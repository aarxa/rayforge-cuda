# Results

GPU: **Tesla T4** (Turing sm_75, 40 SMs) on Google Colab, CUDA 13.0, 2026-11-09.
CPU: Apple **M4 Pro** (8 performance + 4 efficiency cores), clang -O3.
Scenes from RayForge, 1920x1080, `rLimit 5`, `cutOff 1e-4`, one directional light, mirror floor.
GPU times are the best of 10 runs (CUDA events, render kernel only); CPU times are wall clock.

## Render time, 1080p

| Renderer | Dragon (500K tris) | Buddha (868K tris) |
|---|---:|---:|
| Original RayForge, 1 thread (M4 Pro) | 2,800 ms | 5,055 ms |
| Original RayForge, 12 threads (M4 Pro) | 450 ms | 1,004 ms |
| This port on the CPU, 1 thread (M4 Pro) | 272 ms | — |
| This port on the CPU, 12 threads (M4 Pro) | 39 ms | 54 ms |
| **This port on the GPU (T4)** | **18.9 ms** | **21.3 ms** |

- **GPU vs. original RayForge on 12 threads: 24x (dragon), 47x (buddha).**
- Same code, 1 CPU thread, RayForge vs. port: **10x** from data layout and the BVH alone.
- T4 vs. the optimized port on all 12 M4 Pro cores: **2.1x / 2.5x**. Honest takeaway: most of
  the speedup is algorithmic; the T4 (a 70 W, 2018 GPU) adds ~2x over a 2024 laptop CPU.
- 4K (3840x2160) buddha on the T4: **69.9 ms**, 229 Mrays/s.
- Throughput: 188–196 Mrays/s at 1080p (1.8–1.9 rays per pixel incl. reflection and shadow rays).

## Correctness

| Comparison | Dragon | Buddha |
|---|---|---|
| GPU vs. CPU build of the same code | 67.1 dB PSNR, 0.0006% pixels > 2/255 | 63.9 dB, 0.0012% |
| GPU vs. original RayForge render | 51.3 dB PSNR, 0.028% pixels > 2/255 | 54.9 dB, 0.032% |

The ~0.03% of pixels that differ from RayForge sit on the dragon/buddha silhouettes: RayForge's
`ShapeList` returns the first hit in bounding-box order, the port returns the true closest hit,
and the port uses float instead of double. Ground, background and shadows match.

## Ablations (T4, 1080p)

| Variant | Dragon | Buddha | Nodes visited per ray |
|---|---:|---:|---|
| SAH BVH, any-hit shadows (default) | 18.9 ms | 21.3 ms | 4.7 / 6.7 |
| Median-split BVH | 21.6 ms (+14%) | 24.9 ms (+17%) | 15.7 / 18.1 |
| Closest-hit shadow rays | 19.3 ms (+2%) | 20.5 ms (noise) | 4.8 / 6.7 |

SAH cuts BVH node visits ~3x; the time gain is smaller because the GPU is not purely limited by
traversal (recursion and warp divergence cost too). Any-hit shadows barely matter here because
most shadow rays either escape or hit the 2-triangle floor first.

## Why the GPU is "only" ~200 Mrays/s (and what would raise it)
- Recursive `getColor` (`__noinline__`, device stack) limits occupancy and inlining.
- Reflection rays off the mirror floor make warps diverge (some threads bounce, some stop).
- Next steps in NOTES.md: iterative/wavefront shading, ray sorting, 4-wide BVH.
