// Bounding volume hierarchy over triangles, stored as a flat array of 32-byte nodes.
#pragma once
#include <vector>

#include "scene.h"

// Inner node: count == 0, children at leftFirst and leftFirst + 1.
// Leaf: triangles [leftFirst, leftFirst + count) of the (reordered) triangle array.
struct BVHNode {
  V3 bmin;
  int leftFirst;
  V3 bmax;
  int count;
};
static_assert(sizeof(BVHNode) == 32, "BVHNode should be 32 bytes");

enum class SplitMethod { Median, SAH };

struct BVHStats {
  int nodes = 0, leaves = 0, maxDepth = 0;
  double buildSeconds = 0, sahCost = 0;
};

// Builds the BVH and reorders `tris` so each leaf's triangles are contiguous.
std::vector<BVHNode> buildBVH(std::vector<Tri> &tris, SplitMethod method, BVHStats *stats);
