// The ray tracer itself: one code path compiled for both CPU (reference) and GPU.
// Shading follows RayForge's Scene::getColor and DirectionalLight exactly; the differences
// (float math, closest-hit BVH, robust ray offsets) are listed in NOTES.md.
#pragma once
#include "bvh.h"
#include "scene.h"

// Recursive / large functions must not be force-inlined on the device.
#ifdef __CUDACC__
#define HDN __host__ __device__ __noinline__
#else
#define HDN inline
#endif

struct SceneView {  // raw pointers so the same struct works on host and device
  const BVHNode *nodes;
  const Tri *tris;
  const Material *materials;
  const DirLight *lights;
  int numLights;
  Camera camera;
};

struct Ray {
  V3 o, d;
};

struct Hit {
  float t;
  int tri;
  V3 p, n;
};

struct Counters {  // per-thread, summed once at the end (no shared atomics in the hot loop)
  unsigned long long rays = 0, nodeVisits = 0, triTests = 0;
};

// Camera::getRay from RayForge, in float.
HD Ray cameraRay(const Camera &c, int i, int j, int width, int height) {
  const float aspect = (float)width / height;
  const float thetaH = 2.f * atanf(aspect * tanf(c.heightAngle / 2.f));
  const float hh = tanf(c.heightAngle / 2.f), wh = tanf(thetaH / 2.f);
  const float x = (2.f * ((i + 0.5f) / width) - 1.f) * wh;
  const float y = (2.f * ((j + 0.5f) / height) - 1.f) * hh;
  return Ray{c.position, normalize(c.forward + c.right * x + c.up * y)};
}

// Moves a hit point off the surface by a few float ULPs along the normal, on the side the new
// ray leaves from (Wachter and Binder, Ray Tracing Gems ch. 6). RayForge offsets 1e-5 along the
// ray in double; in float at coordinates ~600 that is below one ULP and rays re-hit their own
// triangle ("shadow acne").
HD float offsetComponent(float p, float n) {
  const float kOrigin = 1.f / 32.f, kFloatScale = 1.f / 65536.f, kIntScale = 256.f;
  const int of = (int)(kIntScale * n);
  int bits;
  memcpy(&bits, &p, 4);
  bits += (p < 0) ? -of : of;
  float moved;
  memcpy(&moved, &bits, 4);
  return fabsf(p) < kOrigin ? p + kFloatScale * n : moved;
}
HD V3 offsetRay(V3 p, V3 n, V3 dir) {
  if (dot(n, dir) < 0) n = -n;
  return v3(offsetComponent(p.x, n.x), offsetComponent(p.y, n.y), offsetComponent(p.z, n.z));
}

// Slab test. Returns the entry distance, or kInf if the box is missed or starts beyond tmax.
HD float boxEntry(const BVHNode &b, V3 o, V3 inv, float tmax) {
  const float tx1 = (b.bmin.x - o.x) * inv.x, tx2 = (b.bmax.x - o.x) * inv.x;
  const float ty1 = (b.bmin.y - o.y) * inv.y, ty2 = (b.bmax.y - o.y) * inv.y;
  const float tz1 = (b.bmin.z - o.z) * inv.z, tz2 = (b.bmax.z - o.z) * inv.z;
  const float tnear = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fmaxf(fminf(tz1, tz2), 0.f));
  const float tfar = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fminf(fmaxf(tz1, tz2), tmax));
  return tnear <= tfar ? tnear : kInf;
}

// Moller-Trumbore. Two-sided, like RayForge's plane/barycentric test.
HD float intersectTri(const Tri &t, const Ray &r, float tmax) {
  const V3 p = cross(r.d, t.e2);
  const float det = dot(t.e1, p);
  if (fabsf(det) < 1e-12f) return kInf;
  const float inv = 1.f / det;
  const V3 s = r.o - t.v0;
  const float u = dot(s, p) * inv;
  if (u < 0.f || u > 1.f) return kInf;
  const V3 q = cross(s, t.e1);
  const float v = dot(r.d, q) * inv;
  if (v < 0.f || u + v > 1.f) return kInf;
  const float dist = dot(t.e2, q) * inv;
  return (dist > 0.f && dist < tmax) ? dist : kInf;
}

// Stack-based BVH traversal, nearer child first. With anyHit it returns the first hit found
// (enough for an opaque shadow ray); otherwise the closest hit.
HDN bool traverse(const SceneView &s, const Ray &r, bool anyHit, Hit &hit, Counters &cnt) {
  const V3 inv = v3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
  hit.t = kInf;
  hit.tri = -1;
  cnt.rays++;
  if (boxEntry(s.nodes[0], r.o, inv, kInf) == kInf) return false;
  int stack[64];
  int sp = 0;
  int node = 0;
  while (true) {
    const BVHNode &n = s.nodes[node];
    cnt.nodeVisits++;
    if (n.count > 0) {
      for (int i = n.leftFirst; i < n.leftFirst + n.count; ++i) {
        cnt.triTests++;
        const float t = intersectTri(s.tris[i], r, hit.t);
        if (t < hit.t) {
          hit.t = t;
          hit.tri = i;
          if (anyHit) return true;
        }
      }
      if (sp == 0) break;
      node = stack[--sp];
      continue;
    }
    int a = n.leftFirst, b = n.leftFirst + 1;
    float da = boxEntry(s.nodes[a], r.o, inv, hit.t);
    float db = boxEntry(s.nodes[b], r.o, inv, hit.t);
    if (da > db) {
      const int ti = a; a = b; b = ti;
      const float tf = da; da = db; db = tf;
    }
    if (da == kInf) {
      if (sp == 0) break;
      node = stack[--sp];
    } else {
      node = a;
      if (db != kInf) stack[sp++] = b;
    }
  }
  if (hit.tri >= 0) {
    hit.p = r.o + r.d * hit.t;
    hit.n = s.tris[hit.tri].n;
  }
  return hit.tri >= 0;
}

// DirectionalLight::transparency: march toward the light, multiplying by each blocker's
// transparency. Opaque blockers end the loop at once, so the first hit (any-hit query) is enough.
HDN V3 shadowTransparency(const SceneView &s, const DirLight &l, const Hit &h, V3 cLimit,
                          bool anyHitShadows, Counters &cnt) {
  const V3 L = -l.direction;
  V3 trans = v3(1, 1, 1);
  Ray r{offsetRay(h.p, h.n, L), L};
  Hit b;
  for (int guard = 0; guard < 64; ++guard) {
    if (!traverse(s, r, anyHitShadows, b, cnt)) break;
    V3 kt = s.materials[s.tris[b.tri].material].transparent;
    if (anyHitShadows) {
      if (kt.x == 0 && kt.y == 0 && kt.z == 0) return v3(0, 0, 0);  // opaque: fully shadowed
      // A see-through blocker: the order of blockers matters, so redo this step as closest-hit.
      traverse(s, r, false, b, cnt);
      kt = s.materials[s.tris[b.tri].material].transparent;
    }
    trans = trans * kt;
    if (!allGreater(trans, cLimit)) break;
    r.o = offsetRay(b.p, b.n, L);
  }
  return trans;
}

// Scene::Refract, including its quirks (returns true whenever the x component is a number).
HD bool refractDir(V3 v, V3 n, float ir, V3 &out) {
  const float cosIn = -dot(v, n);
  const float sinIn = sqrtf(fmaxf(0.f, 1.f - cosIn * cosIn));
  float ratio, sinR, cosR;
  if (cosIn < 0) {
    ratio = ir;
    sinR = sinIn * ratio;
    if (sinR > 1) return false;
    cosR = sqrtf(1 - sinR * sinR);
    out = v * ratio + (-n) * (-ratio * cosIn - cosR);
  } else {
    ratio = 1.f / ir;
    sinR = sinIn * ratio;
    if (sinR > 1) return false;
    cosR = sqrtf(1 - sinR * sinR);
    out = v * ratio + n * (ratio * cosIn - cosR);
  }
  return out.x == out.x;  // false only for NaN
}

struct Options {
  int rLimit = 5;
  float cutOff = 1e-4f;
  bool anyHitShadows = true;
};

// Scene::getColor. Note RayForge decrements rDepth on entry and passes rDepth - 1 to children,
// so each bounce costs 2: rLimit 5 traces the primary ray plus 2 bounces.
HDN V3 getColor(const SceneView &s, const Options &o, const Ray &ray, int rDepth, V3 cLimit,
               Counters &cnt) {
  rDepth--;
  Hit h;
  if (!traverse(s, ray, false, h, cnt)) return v3(0, 0, 0);
  const Material &m = s.materials[s.tris[h.tri].material];
  V3 color = m.emissive;
  for (int i = 0; i < s.numLights; ++i) {
    const DirLight &l = s.lights[i];
    color += m.ambient * l.ambient;
    const V3 trans = shadowTransparency(s, l, h, cLimit, o.anyHitShadows, cnt);
    const V3 L = -l.direction;
    const float cosNL = dot(h.n, L);
    if (cosNL > 0) color += m.diffuse * l.diffuse * cosNL * trans;
    const V3 V = normalize(ray.o - h.p);
    const V3 R = (-L) - h.n * (2.f * dot(-L, h.n));
    const float cosVR = dot(V, R);
    if (cosVR > 0) color += m.specular * l.specular * powf(cosVR, m.specularFallOff) * trans;
  }
  const bool canRecurse = rDepth > 0;
  if (canRecurse && allGreater(m.specular, cLimit)) {
    const V3 dir = ray.d - h.n * (2.f * dot(ray.d, h.n));
    const Ray rr{offsetRay(h.p, h.n, dir), dir};
    color += getColor(s, o, rr, rDepth - 1, cLimit / m.specular, cnt) * m.specular;
  }
  V3 tdir;
  if (canRecurse && allGreater(m.transparent, cLimit) && refractDir(ray.d, h.n, m.ir, tdir)) {
    const Ray tr{offsetRay(h.p, h.n, tdir), tdir};
    color += getColor(s, o, tr, rDepth - 1, cLimit / m.transparent, cnt) * m.transparent;
  }
  return clamp01(color);
}

// One pixel, written as 8-bit RGB with RayForge's (int)(c * 255) truncation. Row 0 is the top.
HD void shadePixel(const SceneView &s, const Options &o, int i, int j, int width, int height,
                   unsigned char *rgb, Counters &cnt) {
  const Ray r = cameraRay(s.camera, i, height - j - 1, width, height);
  const V3 c = getColor(s, o, r, o.rLimit, v3(o.cutOff, o.cutOff, o.cutOff), cnt);
  unsigned char *px = rgb + 3 * ((size_t)j * width + i);
  px[0] = (unsigned char)(int)(c.x * 255);
  px[1] = (unsigned char)(int)(c.y * 255);
  px[2] = (unsigned char)(int)(c.z * 255);
}
