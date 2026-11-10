#include "image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
void put16(unsigned char *p, uint16_t v) { p[0] = v & 255; p[1] = v >> 8; }
void put32(unsigned char *p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (v >> (8 * i)) & 255; }
uint32_t get32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
}  // namespace

bool writeBMP(const std::string &path, const Image &img) {
  const int rowBytes = (3 * img.width + 3) & ~3;
  unsigned char h[54] = {'B', 'M'};
  put32(h + 2, 54 + rowBytes * img.height);
  put32(h + 10, 54);
  put32(h + 14, 40);
  put32(h + 18, img.width);
  put32(h + 22, img.height);
  put16(h + 26, 1);
  put16(h + 28, 24);
  put32(h + 34, rowBytes * img.height);
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return false;
  fwrite(h, 1, 54, f);
  std::vector<unsigned char> row(rowBytes, 0);
  for (int y = img.height - 1; y >= 0; --y) {  // BMP rows go bottom-up
    for (int x = 0; x < img.width; ++x) {
      const unsigned char *px = &img.rgb[3 * ((size_t)y * img.width + x)];
      row[3 * x + 0] = px[2];  // BGR
      row[3 * x + 1] = px[1];
      row[3 * x + 2] = px[0];
    }
    fwrite(row.data(), 1, rowBytes, f);
  }
  fclose(f);
  return true;
}

bool readBMP(const std::string &path, Image &img) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  unsigned char h[54];
  if (fread(h, 1, 54, f) != 54 || h[0] != 'B' || h[1] != 'M' || h[28] != 24) { fclose(f); return false; }
  const uint32_t offset = get32(h + 10);
  img.width = (int)get32(h + 18);
  int height = (int)get32(h + 22);
  const bool bottomUp = height > 0;
  img.height = std::abs(height);
  const int rowBytes = (3 * img.width + 3) & ~3;
  img.rgb.assign((size_t)3 * img.width * img.height, 0);
  fseek(f, offset, SEEK_SET);
  std::vector<unsigned char> row(rowBytes);
  for (int r = 0; r < img.height; ++r) {
    if (fread(row.data(), 1, rowBytes, f) != (size_t)rowBytes) { fclose(f); return false; }
    const int y = bottomUp ? img.height - 1 - r : r;
    for (int x = 0; x < img.width; ++x) {
      unsigned char *px = &img.rgb[3 * ((size_t)y * img.width + x)];
      px[0] = row[3 * x + 2];
      px[1] = row[3 * x + 1];
      px[2] = row[3 * x + 0];
    }
  }
  fclose(f);
  return true;
}

Diff compare(const Image &a, const Image &b) {
  Diff d{0, 0, 0};
  if (a.width != b.width || a.height != b.height) { d.psnr = -1; d.pctOver2 = 100; d.maxDiff = 255; return d; }
  double sq = 0;
  size_t over = 0;
  const size_t n = (size_t)a.width * a.height;
  for (size_t i = 0; i < n; ++i) {
    int worst = 0;
    for (int c = 0; c < 3; ++c) {
      const int e = std::abs((int)a.rgb[3 * i + c] - (int)b.rgb[3 * i + c]);
      sq += (double)e * e;
      if (e > worst) worst = e;
    }
    if (worst > 2) over++;
    if (worst > d.maxDiff) d.maxDiff = worst;
  }
  const double mse = sq / (3.0 * n);
  d.psnr = mse == 0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
  d.pctOver2 = 100.0 * over / n;
  return d;
}
