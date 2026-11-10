// 24-bit BMP read/write (the format RayForge outputs) and image comparison.
#pragma once
#include <string>
#include <vector>

struct Image {
  int width = 0, height = 0;
  std::vector<unsigned char> rgb;  // row 0 = top
};

bool writeBMP(const std::string &path, const Image &img);
bool readBMP(const std::string &path, Image &img);

struct Diff {
  double psnr;        // dB; inf if identical
  double pctOver2;    // % of pixels where some channel differs by more than 2/255
  int maxDiff;        // largest per-channel difference
};
Diff compare(const Image &a, const Image &b);
