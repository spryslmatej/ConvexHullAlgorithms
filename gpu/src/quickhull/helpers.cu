#include "helpers.h"

#include "qhData.h"

#include "../utilities/helpers.h"

__device__ double signPosition(const long long p1x, const long long p1y,
                               const long long p2x, const long long p2y,
                               const long long p3x, const long long p3y)
{
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
}

// REARRANGES

__global__ void rearrangeFlags(qhData data, FLAG_TYPE *help)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    help[thIndex] = data.rawFlag[data.rawIndexes[thIndex]];
}

__global__ void rearrangeX(qhData data, COORDS_TYPE *help)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    help[thIndex] = data.rawX[data.rawIndexes[thIndex]];
}

__global__ void rearrangeY(qhData data, COORDS_TYPE *help)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    help[thIndex] = data.rawY[data.rawIndexes[thIndex]];
}

__global__ void rearrangeKeys(qhData data, KEYS_TYPE *keysHelp)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    keysHelp[thIndex] = data.rawKeys[data.rawIndexes[thIndex]];
}

__global__ void rearrangeDistances(qhData data, DISTANCES_TYPE *distancesHelp)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    distancesHelp[thIndex] = data.rawDistances[data.rawIndexes[thIndex]];
}

__global__ void rearrangeHeads(qhData data, HEAD_TYPE *help)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    help[thIndex] = data.rawHead[data.rawIndexes[thIndex]];
}

__global__ void rearrangeIndexes(qhData data, INDEXES_TYPE *help)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    help[thIndex] = data.rawIndexes[thIndex];
}
