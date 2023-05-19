#pragma once

#include "qhData.h"

__device__ double signPosition(const long long p1x, const long long p1y,
                               const long long p2x, const long long p2y,
                               const long long p3x, const long long p3y);

// REARRANGES

__global__ void rearrangeFlags(qhData devData, FLAG_TYPE *help);

__global__ void rearrangeX(qhData devData, COORDS_TYPE *help);

__global__ void rearrangeY(qhData devData, COORDS_TYPE *help);

__global__ void rearrangeKeys(qhData devData, KEYS_TYPE *keysHelp);

__global__ void rearrangeDistances(qhData devData, DISTANCES_TYPE *distancesHelp);

__global__ void rearrangeHeads(qhData devData, HEAD_TYPE *help);

__global__ void rearrangeIndexes(qhData devData, INDEXES_TYPE *help);
