// Command-line options shared by the CPU and GPU drivers.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bvh.h"
#include "trace.h"

struct Args {
  std::string in, out, ref;
  int width = 1920, height = 1080, threads = 0, reps = 5;
  SplitMethod split = SplitMethod::SAH;
  Options opts;
  bool cpuCheck = false;  // GPU driver only: also render on the CPU and compare
};

inline Args parseArgs(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const char *k = argv[i];
    const char *v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(k, "--in")) a.in = v, ++i;
    else if (!strcmp(k, "--out")) a.out = v, ++i;
    else if (!strcmp(k, "--ref")) a.ref = v, ++i;
    else if (!strcmp(k, "--width")) a.width = atoi(v), ++i;
    else if (!strcmp(k, "--height")) a.height = atoi(v), ++i;
    else if (!strcmp(k, "--threads")) a.threads = atoi(v), ++i;
    else if (!strcmp(k, "--reps")) a.reps = atoi(v), ++i;
    else if (!strcmp(k, "--rLimit")) a.opts.rLimit = atoi(v), ++i;
    else if (!strcmp(k, "--cutOff")) a.opts.cutOff = (float)atof(v), ++i;
    else if (!strcmp(k, "--bvh")) a.split = !strcmp(v, "median") ? SplitMethod::Median : SplitMethod::SAH, ++i;
    else if (!strcmp(k, "--shadows")) a.opts.anyHitShadows = strcmp(v, "closest") != 0, ++i;
    else if (!strcmp(k, "--cpu-check")) a.cpuCheck = true;
    else { fprintf(stderr, "unknown option %s\n", k); exit(2); }
  }
  if (a.in.empty()) { fprintf(stderr, "usage: %s --in scene.ray [options]\n", argv[0]); exit(2); }
  return a;
}
