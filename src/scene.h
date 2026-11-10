// Scene data in flat arrays, ready to copy to the GPU.
#pragma once
#include <string>
#include <vector>

#include "common.h"

struct Camera {
  V3 position, forward, up, right;
  float heightAngle;  // radians
};

struct DirLight {
  V3 ambient, diffuse, specular, direction;  // direction is unit, pointing away from the light
};

struct Material {
  V3 emissive, ambient, diffuse, specular;
  float specularFallOff;
  V3 transparent;
  float ir;
};

// Triangle stored as v0 plus two edges (what Moller-Trumbore needs) and a flat unit normal,
// matching RayForge's Triangle::intersect, which shades with the face normal.
struct Tri {
  V3 v0, e1, e2, n;
  int material;
};

struct Scene {
  Camera camera;
  std::vector<DirLight> lights;
  std::vector<Material> materials;
  std::vector<Tri> tris;
};

// Parses the flattened RayForge .ray format used by the mesh scenes (dragon, buddha, bunny):
// #camera, #light_dir, #material, #vertex, #shape_triangles <material>, #shape_triangle a b c.
// The file's own #shape_list nesting is ignored; we build our own BVH. Unsupported
// directives (other light or shape types, textures, transforms) stop with an error.
Scene loadRayFile(const std::string &path);
