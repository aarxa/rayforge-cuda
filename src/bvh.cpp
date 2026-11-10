#include "bvh.h"

#include <algorithm>
#include <chrono>

namespace {

struct AABB {
  V3 lo = v3(kInf, kInf, kInf), hi = v3(-kInf, -kInf, -kInf);
  void grow(V3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
  void grow(const AABB &b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
  float area() const {
    const V3 d = hi - lo;
    return (d.x < 0) ? 0.f : 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
  }
};

struct Prim {
  AABB box;
  V3 centroid;
  int index;
};

constexpr int kBins = 16;
constexpr int kMaxLeaf = 8;     // never put more than this many triangles in a leaf
constexpr float kTraversal = 1.0f;  // SAH cost of visiting a node, relative to one triangle test

struct Builder {
  std::vector<Prim> prims;
  std::vector<BVHNode> nodes;
  SplitMethod method;
  BVHStats *stats;

  void build(int nodeIndex, int first, int count, int depth) {
    AABB box, cbox;
    for (int i = first; i < first + count; ++i) {
      box.grow(prims[i].box);
      cbox.grow(prims[i].centroid);
    }
    BVHNode &node = nodes[nodeIndex];
    node.bmin = box.lo;
    node.bmax = box.hi;
    stats->maxDepth = std::max(stats->maxDepth, depth);

    int mid = -1;
    if (count > 1) mid = method == SplitMethod::SAH ? splitSAH(box, cbox, first, count)
                                                    : splitMedian(cbox, first, count);
    if (mid < 0) {  // make a leaf
      node.leftFirst = first;
      node.count = count;
      stats->leaves++;
      return;
    }
    const int left = (int)nodes.size();
    nodes.push_back({});
    nodes.push_back({});
    nodes[nodeIndex].leftFirst = left;  // re-index: push_back may have reallocated
    nodes[nodeIndex].count = 0;
    build(left, first, mid - first, depth + 1);
    build(left + 1, mid, first + count - mid, depth + 1);
  }

  // Splits at the centroid median along the widest axis. Returns the split index, or -1 for a leaf.
  int splitMedian(const AABB &cbox, int first, int count) {
    if (count <= 2) return -1;
    const V3 ext = cbox.hi - cbox.lo;
    const int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    const int mid = first + count / 2;
    std::nth_element(prims.begin() + first, prims.begin() + mid, prims.begin() + first + count,
                     [axis](const Prim &a, const Prim &b) {
                       return comp(a.centroid, axis) < comp(b.centroid, axis);
                     });
    return mid;
  }

  // Binned surface area heuristic on all three axes. Returns -1 if a leaf is cheaper.
  int splitSAH(const AABB &box, const AABB &cbox, int first, int count) {
    float bestCost = kInf;
    int bestAxis = -1, bestSplit = -1;
    for (int axis = 0; axis < 3; ++axis) {
      const float lo = comp(cbox.lo, axis), hi = comp(cbox.hi, axis);
      if (hi <= lo) continue;
      AABB bins[kBins];
      int counts[kBins] = {0};
      const float scale = kBins / (hi - lo);
      for (int i = first; i < first + count; ++i) {
        const int b = std::min(kBins - 1, (int)((comp(prims[i].centroid, axis) - lo) * scale));
        bins[b].grow(prims[i].box);
        counts[b]++;
      }
      // Sweep from the right to get the area/count of every right side, then from the left.
      float rightArea[kBins];
      int rightCount[kBins];
      AABB acc;
      int n = 0;
      for (int b = kBins - 1; b > 0; --b) {
        acc.grow(bins[b]);
        n += counts[b];
        rightArea[b] = acc.area();
        rightCount[b] = n;
      }
      acc = AABB();
      n = 0;
      for (int b = 0; b < kBins - 1; ++b) {
        acc.grow(bins[b]);
        n += counts[b];
        if (n == 0 || rightCount[b + 1] == 0) continue;
        const float cost = n * acc.area() + rightCount[b + 1] * rightArea[b + 1];
        if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = b; }
      }
    }
    const float parentArea = box.area();
    const float splitCost = kTraversal + (parentArea > 0 ? bestCost / parentArea : kInf);
    // All centroids equal: no SAH split exists, so fall back to splitting the list in half.
    if (bestAxis < 0) return count <= kMaxLeaf ? -1 : splitMedian(cbox, first, count);
    if (splitCost >= count && count <= kMaxLeaf) return -1;
    const float lo = comp(cbox.lo, bestAxis), hi = comp(cbox.hi, bestAxis);
    const float scale = kBins / (hi - lo);
    auto it = std::partition(prims.begin() + first, prims.begin() + first + count, [&](const Prim &p) {
      return std::min(kBins - 1, (int)((comp(p.centroid, bestAxis) - lo) * scale)) <= bestSplit;
    });
    const int mid = (int)(it - prims.begin());
    if (mid == first || mid == first + count) return splitMedian(cbox, first, count);
    return mid;
  }

  double sah(int nodeIndex) const {
    const BVHNode &n = nodes[nodeIndex];
    const V3 d = n.bmax - n.bmin;
    const double area = 2.0 * ((double)d.x * d.y + (double)d.y * d.z + (double)d.z * d.x);
    if (n.count > 0) return area * n.count;
    return area * kTraversal + sah(n.leftFirst) + sah(n.leftFirst + 1);
  }
};

}  // namespace

std::vector<BVHNode> buildBVH(std::vector<Tri> &tris, SplitMethod method, BVHStats *stats) {
  const auto t0 = std::chrono::steady_clock::now();
  Builder b;
  b.method = method;
  b.stats = stats;
  b.prims.resize(tris.size());
  for (size_t i = 0; i < tris.size(); ++i) {
    const Tri &t = tris[i];
    Prim &p = b.prims[i];
    p.box.grow(t.v0);
    p.box.grow(t.v0 + t.e1);
    p.box.grow(t.v0 + t.e2);
    p.centroid = (p.box.lo + p.box.hi) * 0.5f;
    p.index = (int)i;
  }
  b.nodes.reserve(2 * tris.size());
  b.nodes.push_back({});
  b.build(0, 0, (int)tris.size(), 0);

  std::vector<Tri> ordered(tris.size());
  for (size_t i = 0; i < tris.size(); ++i) ordered[i] = tris[b.prims[i].index];
  tris.swap(ordered);

  stats->nodes = (int)b.nodes.size();
  const BVHNode &root = b.nodes[0];
  const V3 d = root.bmax - root.bmin;
  stats->sahCost = b.sah(0) / (2.0 * ((double)d.x * d.y + (double)d.y * d.z + (double)d.z * d.x));
  stats->buildSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return b.nodes;
}
