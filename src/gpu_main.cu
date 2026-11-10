// GPU renderer: one CUDA thread per pixel, running the same trace.h code as the CPU build.
// Usage: ./rayforge_gpu --in scene.ray [--width 1920] [--height 1080] [--reps 5] [--out out.bmp]
//                       [--ref rayforge.bmp] [--cpu-check] [--bvh sah|median] [--shadows any|closest]
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "args.h"
#include "image.h"
#include "trace.h"

#define CUDA_CHECK(x)                                                                        \
  do {                                                                                       \
    cudaError_t err_ = (x);                                                                  \
    if (err_ != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

constexpr int kBlockX = 8, kBlockY = 8;  // 8x8 tiles keep neighbouring rays in the same warp

// kStats adds a warp-level reduction of the per-thread counters and one atomicAdd per warp
// (not per ray), so counting does not serialize the GPU the way a shared counter would.
template <bool kStats>
__global__ void renderKernel(SceneView s, Options o, int width, int height, unsigned char *rgb,
                             unsigned long long *stats) {
  const int i = blockIdx.x * kBlockX + threadIdx.x;
  const int j = blockIdx.y * kBlockY + threadIdx.y;
  Counters cnt;
  if (i < width && j < height) shadePixel(s, o, i, j, width, height, rgb, cnt);
  if (kStats) {
    unsigned long long v[3] = {cnt.rays, cnt.nodeVisits, cnt.triTests};
    for (int k = 0; k < 3; ++k) {
      for (int off = 16; off > 0; off >>= 1) v[k] += __shfl_down_sync(0xffffffffu, v[k], off);
    }
    const int lane = (threadIdx.y * kBlockX + threadIdx.x) & 31;
    if (lane == 0)
      for (int k = 0; k < 3; ++k) atomicAdd(&stats[k], v[k]);
  }
}

template <typename T>
T *upload(const std::vector<T> &v) {
  T *d;
  CUDA_CHECK(cudaMalloc(&d, sizeof(T) * v.size()));
  CUDA_CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  return d;
}

int main(int argc, char **argv) {
  Args a = parseArgs(argc, argv);
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  const auto t0 = std::chrono::steady_clock::now();
  Scene scene = loadRayFile(a.in);
  const double loadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  BVHStats bs;
  std::vector<BVHNode> nodes = buildBVH(scene.tris, a.split, &bs);
  printf("scene %s: %zu triangles (load %.2f s); BVH (%s): %d nodes, depth %d, SAH cost %.1f, build %.2f s\n",
         a.in.c_str(), scene.tris.size(), loadS, a.split == SplitMethod::SAH ? "SAH" : "median",
         bs.nodes, bs.maxDepth, bs.sahCost, bs.buildSeconds);

  // getColor recurses (reflection/refraction) and keeps a 64-entry traversal stack per call.
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

  const auto u0 = std::chrono::steady_clock::now();
  SceneView dev{upload(nodes), upload(scene.tris), upload(scene.materials), upload(scene.lights),
                (int)scene.lights.size(), scene.camera};
  unsigned char *dRGB;
  const size_t bytes = (size_t)3 * a.width * a.height;
  CUDA_CHECK(cudaMalloc(&dRGB, bytes));
  unsigned long long *dStats;
  CUDA_CHECK(cudaMalloc(&dStats, 3 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaDeviceSynchronize());
  const double uploadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - u0).count();
  printf("upload to GPU: %.1f MB in %.3f s\n",
         (nodes.size() * sizeof(BVHNode) + scene.tris.size() * sizeof(Tri)) / 1e6, uploadS);

  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((a.width + kBlockX - 1) / kBlockX, (a.height + kBlockY - 1) / kBlockY);

  // Stats pass (also warms up the GPU).
  CUDA_CHECK(cudaMemset(dStats, 0, 3 * sizeof(unsigned long long)));
  renderKernel<true><<<grid, block>>>(dev, a.opts, a.width, a.height, dRGB, dStats);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long st[3];
  CUDA_CHECK(cudaMemcpy(st, dStats, sizeof(st), cudaMemcpyDeviceToHost));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float bestMs = 1e30f, sumMs = 0;
  for (int r = 0; r < a.reps; ++r) {
    CUDA_CHECK(cudaEventRecord(e0));
    renderKernel<false><<<grid, block>>>(dev, a.opts, a.width, a.height, dRGB, nullptr);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    bestMs = fminf(bestMs, ms);
    sumMs += ms;
  }
  CUDA_CHECK(cudaGetLastError());
  printf("GPU render %dx%d: best %.2f ms, mean %.2f ms over %d runs, %.0f Mrays/s "
         "(%.2f rays/px, %.1f nodes/ray, %.1f tris/ray)\n",
         a.width, a.height, bestMs, sumMs / a.reps, a.reps, st[0] / (bestMs * 1e3),
         (double)st[0] / (a.width * a.height), (double)st[1] / st[0], (double)st[2] / st[0]);

  Image img;
  img.width = a.width;
  img.height = a.height;
  img.rgb.resize(bytes);
  CUDA_CHECK(cudaMemcpy(img.rgb.data(), dRGB, bytes, cudaMemcpyDeviceToHost));
  if (!a.out.empty() && !writeBMP(a.out, img)) fprintf(stderr, "could not write %s\n", a.out.c_str());

  if (a.cpuCheck) {  // same code on the CPU: should match the GPU image almost exactly
    SceneView host{nodes.data(), scene.tris.data(), scene.materials.data(), scene.lights.data(),
                   (int)scene.lights.size(), scene.camera};
    Image cpu = img;
    const int nThreads = (int)std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> nextRow(0);
    const auto c0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t)
      pool.emplace_back([&] {
        Counters c;
        for (int j = nextRow++; j < a.height; j = nextRow++)
          for (int i = 0; i < a.width; ++i) shadePixel(host, a.opts, i, j, a.width, a.height, cpu.rgb.data(), c);
      });
    for (auto &th : pool) th.join();
    const double cpuS = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    const Diff d = compare(img, cpu);
    printf("CPU check (%d threads, %.3f s): GPU vs CPU PSNR %.2f dB, %.4f%% pixels differ by > 2/255, max diff %d\n",
           nThreads, cpuS, d.psnr, d.pctOver2, d.maxDiff);
  }
  if (!a.ref.empty()) {
    Image ref;
    if (!readBMP(a.ref, ref)) { fprintf(stderr, "could not read %s\n", a.ref.c_str()); return 1; }
    const Diff d = compare(img, ref);
    printf("vs RayForge reference %s: PSNR %.2f dB, %.3f%% pixels differ by > 2/255, max diff %d\n",
           a.ref.c_str(), d.psnr, d.pctOver2, d.maxDiff);
  }
  return 0;
}
