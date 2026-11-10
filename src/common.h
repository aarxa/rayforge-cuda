// Small vector type and helpers shared by the CPU and GPU builds.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

struct V3 {
  float x, y, z;
};

HD V3 v3(float x, float y, float z) { return V3{x, y, z}; }
HD V3 operator+(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD V3 operator-(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD V3 operator-(V3 a) { return v3(-a.x, -a.y, -a.z); }
HD V3 operator*(V3 a, V3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
HD V3 operator*(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
HD V3 operator/(V3 a, V3 b) { return v3(a.x / b.x, a.y / b.y, a.z / b.z); }
HD V3 &operator+=(V3 &a, V3 b) { a = a + b; return a; }
HD float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD V3 cross(V3 a, V3 b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
HD float length(V3 a) { return sqrtf(dot(a, a)); }
HD V3 normalize(V3 a) { return a * (1.f / length(a)); }
HD V3 vmin(V3 a, V3 b) { return v3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
HD V3 vmax(V3 a, V3 b) { return v3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
HD float comp(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
HD bool allGreater(V3 a, V3 b) { return a.x > b.x && a.y > b.y && a.z > b.z; }
HD V3 clamp01(V3 c) {
  return v3(fminf(fmaxf(c.x, 0.f), 1.f), fminf(fmaxf(c.y, 0.f), 1.f), fminf(fmaxf(c.z, 0.f), 1.f));
}

constexpr float kInf = 1e30f;
