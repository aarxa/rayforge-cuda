#include "scene.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

V3 readV3(std::istream &in) {
  V3 v;
  in >> v.x >> v.y >> v.z;
  return v;
}

[[noreturn]] void fail(const std::string &msg) { throw std::runtime_error(msg); }

}  // namespace

Scene loadRayFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) fail("cannot open " + path);
  // Read the whole file into memory: the mesh scenes are 50-100 MB of text.
  std::stringstream in;
  in << file.rdbuf();

  Scene s;
  std::vector<V3> positions;
  int currentMaterial = -1;
  bool haveCamera = false;
  std::string word;
  while (in >> word) {
    if (word[0] != '#') fail("unexpected token '" + word + "' in " + path);
    const std::string key = word.substr(1);
    if (key == "vertex") {
      V3 p = readV3(in);
      readV3(in);  // per-vertex normal: unused, RayForge shades triangles with the face normal
      float u, v;
      in >> u >> v;
      positions.push_back(p);
    } else if (key == "shape_triangle") {
      int a, b, c;
      in >> a >> b >> c;
      if (currentMaterial < 0) fail("#shape_triangle outside #shape_triangles");
      const int n = (int)positions.size();
      if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) fail("vertex index out of range");
      Tri t;
      t.v0 = positions[a];
      t.e1 = positions[b] - positions[a];
      t.e2 = positions[c] - positions[a];
      // Normal in double, like RayForge, then stored as float.
      const double nx = (double)t.e1.y * t.e2.z - (double)t.e1.z * t.e2.y;
      const double ny = (double)t.e1.z * t.e2.x - (double)t.e1.x * t.e2.z;
      const double nz = (double)t.e1.x * t.e2.y - (double)t.e1.y * t.e2.x;
      const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len == 0) continue;  // degenerate: RayForge can never hit it either
      t.n = v3((float)(nx / len), (float)(ny / len), (float)(nz / len));
      t.material = currentMaterial;
      s.tris.push_back(t);
    } else if (key == "shape_list_begin" || key == "shape_list_end") {
      // RayForge's precomputed hierarchy; ignored.
    } else if (key == "shape_triangles") {
      in >> currentMaterial;
      if (currentMaterial < 0 || currentMaterial >= (int)s.materials.size())
        fail("material index out of range");
    } else if (key == "material") {
      Material m;
      m.emissive = readV3(in);
      m.ambient = readV3(in);
      m.diffuse = readV3(in);
      m.specular = readV3(in);
      in >> m.specularFallOff;
      m.transparent = readV3(in);
      in >> m.ir;
      int tex;
      std::string name;
      in >> tex >> name;
      if (tex >= 0) fail("textures are not supported yet");
      s.materials.push_back(m);
    } else if (key == "light_dir") {
      DirLight l;
      l.ambient = readV3(in);
      l.diffuse = readV3(in);
      l.specular = readV3(in);
      l.direction = normalize(readV3(in));
      s.lights.push_back(l);
    } else if (key == "camera") {
      Camera &c = s.camera;
      c.position = readV3(in);
      c.forward = readV3(in);
      c.up = normalize(readV3(in));
      in >> c.heightAngle;
      if (c.heightAngle > 3.14159265f) c.heightAngle *= 3.14159265f / 180.f;  // degrees
      c.right = cross(c.forward, c.up);
      haveCamera = true;
    } else {
      fail("unsupported directive #" + key + " (only mesh scenes with directional lights)");
    }
    if (!in && !in.eof()) fail("parse error after #" + key);
  }
  if (!haveCamera) fail("no #camera in " + path);
  if (s.tris.empty()) fail("no triangles in " + path);
  return s;
}
