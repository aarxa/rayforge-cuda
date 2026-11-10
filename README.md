# rayforge-cuda

A CUDA port of [RayForge](https://github.com/aarxa/RayForge-CPP-Ray-Tracing-Engine), my C++ ray
tracer: the same shading model (Phong lighting, shadow-ray transparency, recursive reflection and
refraction) running one CUDA thread per pixel over a binned-SAH BVH, checked pixel by pixel against
RayForge's own output.

## Results
On a Tesla T4, 1080p: dragon (500K triangles) in **18.9 ms**, buddha (868K) in **21.3 ms**: **24x and 47x faster** than the original RayForge on 12 M4 Pro cores, with 0.03% of pixels differing from its output. About 10x of that comes from the BVH and data layout (same CPU, 1 thread); the GPU adds ~2x over the optimized CPU port. Full tables: [RESULTS.md](RESULTS.md).

## How it works
- `src/trace.h`: the ray tracer, written once as `__host__ __device__` code and compiled for both
  CPU and GPU, so the GPU image can be checked against a CPU run of the identical code.
- `src/bvh.cpp`: binned surface-area-heuristic BVH (or median split, `--bvh median`) in a flat array
  of 32-byte nodes.
- `src/gpu_main.cu`: uploads the scene, renders with 8x8 thread blocks, times with CUDA events.
- `src/cpu_main.cpp`: the same renderer on `std::thread`, used as the CPU baseline.
- Design choices and what changed from RayForge: [NOTES.md](NOTES.md).

## Run it
No NVIDIA GPU? Open `colab.ipynb` in Google Colab with a T4 runtime and Run all.

```bash
make rayforge_gpu rayforge_cpu
./rayforge_gpu --in dragon.ray --ref reference/dragon_cpu_1080p.bmp --cpu-check --out dragon.bmp
./rayforge_cpu --in dragon.ray --threads 12
```
Scenes (`dragon.ray`, `buddha.ray`) come from the RayForge repo. `reference/` holds RayForge's
own 1920x1080 renders.

## Scope
Supports the mesh scenes (triangles, materials, directional lights). Point/spot lights, textures
and RayForge's analytic shapes (sphere, cone, torus...) are not ported yet; the loader stops with a
clear error instead of rendering something different.
