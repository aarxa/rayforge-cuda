// CPU reference renderer: the same trace.h code, run with std::thread (one row at a time from
// an atomic counter, like RayForge). Used to check the GPU output and as a CPU baseline.
// Usage: ./rayforge_cpu --in scene.ray [--width 1920] [--height 1080] [--out out.bmp]
//                       [--ref reference.bmp] [--threads N] [--bvh sah|median] [--shadows any|closest]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "args.h"
#include "image.h"
#include "trace.h"

int main(int argc, char **argv) {
  Args a = parseArgs(argc, argv);
  const auto t0 = std::chrono::steady_clock::now();
  Scene scene = loadRayFile(a.in);
  const double loadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  BVHStats bs;
  std::vector<BVHNode> nodes = buildBVH(scene.tris, a.split, &bs);
  printf("scene %s: %zu triangles, %zu materials, %zu lights (load %.2f s)\n", a.in.c_str(),
         scene.tris.size(), scene.materials.size(), scene.lights.size(), loadS);
  printf("BVH (%s): %d nodes, %d leaves, depth %d, SAH cost %.1f, build %.2f s\n",
         a.split == SplitMethod::SAH ? "SAH" : "median", bs.nodes, bs.leaves, bs.maxDepth,
         bs.sahCost, bs.buildSeconds);

  SceneView view{nodes.data(), scene.tris.data(), scene.materials.data(), scene.lights.data(),
                 (int)scene.lights.size(), scene.camera};
  Image img;
  img.width = a.width;
  img.height = a.height;
  img.rgb.assign((size_t)3 * a.width * a.height, 0);

  const int nThreads = a.threads > 0 ? a.threads : (int)std::max(1u, std::thread::hardware_concurrency());
  std::vector<Counters> counters(nThreads);
  std::atomic<int> nextRow(0);
  const auto r0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < nThreads; ++t)
    pool.emplace_back([&, t] {
      for (int j = nextRow++; j < a.height; j = nextRow++)
        for (int i = 0; i < a.width; ++i)
          shadePixel(view, a.opts, i, j, a.width, a.height, img.rgb.data(), counters[t]);
    });
  for (auto &th : pool) th.join();
  const double renderS = std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();

  Counters total;
  for (auto &c : counters) { total.rays += c.rays; total.nodeVisits += c.nodeVisits; total.triTests += c.triTests; }
  printf("CPU render %dx%d on %d threads: %.3f s, %.1f Mrays/s (%.2f rays/px, %.1f nodes/ray, %.1f tris/ray)\n",
         a.width, a.height, nThreads, renderS, total.rays / renderS / 1e6,
         (double)total.rays / (a.width * a.height), (double)total.nodeVisits / total.rays,
         (double)total.triTests / total.rays);

  if (!a.out.empty() && !writeBMP(a.out, img)) fprintf(stderr, "could not write %s\n", a.out.c_str());
  if (!a.ref.empty()) {
    Image ref;
    if (!readBMP(a.ref, ref)) { fprintf(stderr, "could not read %s\n", a.ref.c_str()); return 1; }
    const Diff d = compare(img, ref);
    printf("vs %s: PSNR %.2f dB, %.3f%% of pixels differ by > 2/255, max diff %d\n",
           a.ref.c_str(), d.psnr, d.pctOver2, d.maxDiff);
  }
  return 0;
}
