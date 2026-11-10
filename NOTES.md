# Design notes: porting RayForge to CUDA

Read this before talking about the project. Each section is a likely interview question.

## 1. What was ported
RayForge's `Scene::getColor` recursion and `DirectionalLight` shading, line for line
(`src/trace.h`): emissive + ambient, shadow-ray transparency loop, Lambert diffuse, Phong
specular, reflection and refraction with the same `rLimit` / `cutOff` rules, and RayForge's
`(int)(c * 255)` pixel conversion. Even its quirks are kept so the images can be compared:
`rDepth` drops by 2 per bounce (decrement on entry, then `rDepth - 1`), and `Refract` only checks
the x component for NaN.

## 2. What had to change, and why
| RayForge (CPU) | Port | Why |
|---|---|---|
| `double` everywhere | `float` | T4 FP64 throughput is 1/32 of FP32 |
| 1e-5 offset along the new ray | ULP-based offset along the normal (Wachter & Binder, *Ray Tracing Gems* ch. 6) | in float at coordinates ~600, 1e-5 is below one ULP, so rays re-hit their own triangle (shadow acne) |
| Scene-graph `ShapeList`s, `std::vector` + `std::sort` of child hits per node, `std::function` per test | one flat BVH array, explicit 64-entry stack, nearer child first | no heap, no virtual calls, no lambdas: all of these are unavailable or slow on a GPU |
| `ShapeList` returns the first hit in box-entry order | true closest hit (tmax shrinks as hits are found) | first-hit can return a farther triangle when boxes overlap; this is the main source of the ~0.03% differing pixels |
| shadow rays find the closest blocker | any-hit for opaque blockers, closest only for see-through ones | a shadow ray only needs to know *whether* something is in the way |
| `std::atomic` / `thread_local` stats | per-thread counters, warp shuffle reduction, one `atomicAdd` per warp | same lesson as RayForge's 4.4x atomic slowdown, applied to 2 million GPU threads |

## 3. The BVH
- **Binned SAH** (16 bins, 3 axes): split cost = traversal + (N_L·A_L + N_R·A_R)/A; make a leaf when
  that is not cheaper than testing every triangle. Compare with `--bvh median`.
- **32-byte nodes** (`BVHNode`): two nodes per 64-byte cache line; children are stored next to each
  other, so a node only needs one index.
- **Triangles reordered** so each leaf's triangles are contiguous in memory.

## 4. Why one thread per pixel, 8x8 blocks
Neighbouring pixels shoot similar rays, which visit the same BVH nodes. A warp built from an 8x4
patch diverges less than one built from a 32-pixel row. Divergence (threads in a warp taking
different branches) is the main cost of ray tracing on GPUs: a warp runs until its slowest ray.

## 5. Recursion on the GPU
CUDA allows recursion but each thread needs stack memory, so `main` raises `cudaLimitStackSize`.
`getColor` and `traverse` are `__noinline__` to keep register use and code size down.
A production renderer would convert this to a loop ("wavefront" path tracing, as in OptiX); a good
next step and a good interview discussion.

## 6. How correctness is checked
1. The CPU build of the same code is compared with RayForge's own renders (PSNR, % differing pixels).
2. On the GPU, `--cpu-check` renders the same frame with the same code on the CPU and compares.
3. Differences are counted per pixel, not eyeballed, and located (they sit on the dragon's thin edges).

## 7. Where the speed comes from (be honest about this in interviews)
- RayForge 1 thread → port 1 thread (same CPU): about **10x**, all from data layout and the BVH.
- Port, 12 M4 Pro threads → T4 GPU: about **2x** (39 → 18.9 ms on dragon). See RESULTS.md.
- Original RayForge, 12 threads → T4: **24x / 47x** (dragon / buddha).
Saying "most of my first 10x was algorithmic" is a strong answer; it shows you measured.

## Next steps
- Wavefront / iterative `getColor` (no recursion, better occupancy)
- BVH with 4-wide nodes or a compressed layout; sort rays by direction for coherence
- Point and spot lights, textures (RayForge supports them; the loader rejects them today)
- Profile with Nsight Compute: achieved occupancy, warp execution efficiency, L2 hit rate
