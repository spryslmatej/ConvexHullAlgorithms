#include "firstSplit.h"

#include <iostream>
#include <cmath>

#include "../model/point.h"
#include "../utilities/handleError.cu"
#include "../utilities/helpers.h"

#include "qhData.h"
#include "helpers.h"
#include "partition.h"
#include "cPoint.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>

using namespace std;

struct compareCoords
{
    COORDS_TYPE *rawX, *rawY;
    compareCoords(COORDS_TYPE *rx, COORDS_TYPE *ry) : rawX(rx), rawY(ry) {}

    __device__ bool operator()(const INDEXES_TYPE i,
                               const INDEXES_TYPE j)
    {
        return (rawX[i] < rawX[j]) ||
               (rawX[i] == rawX[j] && rawY[i] < rawY[j]);
    }
};

void CFirstSplit::findExtremePoints(const qhData data, unsigned &minIndex, unsigned &maxIndex)
{
    thrust::device_vector<COORDS_TYPE> xHelp(data.ptsSize);
    thrust::device_vector<COORDS_TYPE> yHelp(data.ptsSize);

    COORDS_TYPE *rawX = thrust::raw_pointer_cast(xHelp.data());
    COORDS_TYPE *rawY = thrust::raw_pointer_cast(yHelp.data());

    rearrangeX<<<blocks, threads>>>(data, rawX);
    rearrangeY<<<blocks, threads>>>(data, rawY);

    auto extremas =
        thrust::minmax_element(
            thrust::device,
            data.devIndexes.begin(),
            data.devIndexes.end(),
            compareCoords(rawX, rawY));

    minIndex = thrust::distance(data.devIndexes.begin(), extremas.first);
    maxIndex = thrust::distance(data.devIndexes.begin(), extremas.second);
}

__global__ void updateKeys(qhData data, unsigned split)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myIndex = data.rawIndexes[thIndex];

    data.rawKeys[myIndex] = thIndex < split ? 0 : 1;
}

__global__ void assignFlagsToPoints(qhData data, unsigned minIndex, unsigned maxIndex)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myIndex = data.rawIndexes[thIndex];

    double val = signPosition(data.rawX[minIndex], data.rawY[minIndex],
                              data.rawX[maxIndex], data.rawY[maxIndex],
                              data.rawX[myIndex], data.rawY[myIndex]);

    data.rawFlag[myIndex] = val > 0 ? 1 : 0;
}

__global__ void updateIndexes(qhData data, cPoint *points)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myIndex = data.rawIndexes[thIndex];

    data.rawIndexes[myIndex] = points[myIndex].ptIndex;
}

__global__ void arraysToPoints(qhData data, cPoint *points)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    INDEXES_TYPE index = data.rawIndexes[thIndex];

    COORDS_TYPE x = data.rawX[index],
                y = data.rawY[index];

    points[thIndex] = cPoint(x, y, index);
}

unsigned CFirstSplit::partitionPoints(qhData &data)
{
    unsigned min, max;
    findExtremePoints(data, min, max);

    data.devHead[min] = 1;
    data.devHead[max] = 1;

    data.devFirstPts[0] = min;
    data.devFirstPts[1] = max;
    data.rawFirstPtsSize = 2;

    assignFlagsToPoints<<<blocks, threads>>>(data, min, max);

    // manually set flag for max point
    data.devFlag[max] = 1;

    // prepare flags for partition
    rearrangeFlags<<<blocks, threads>>>(data,
                                        thrust::raw_pointer_cast(flagsHelp.data()));

    // partition points by flags
    auto splittingPoint = thrust::partition(
        thrust::device,
        data.devIndexes.begin(),
        data.devIndexes.end(),
        flagsHelp.begin(),
        partitionByFlag());

    // split index refers to the print-ordering
    unsigned splitIndex = thrust::distance(data.devIndexes.begin(), splittingPoint);

    // cout << "lower: <0," << splitIndex
    //      << ") ["
    //      << thrust::distance(data.rawIndexes, data.rawIndexes + splitIndex)
    //      << "], upper: <"
    //      << splitIndex << "," << data.ptsSize
    //      << ") ["
    //      << thrust::distance(data.rawIndexes + splitIndex, data.rawIndexes + data.ptsSize)
    //      << "]" << endl;

    return splitIndex;
}
void CFirstSplit::sortPartitions(qhData &data, unsigned splitIndex)
{
    rearrangeX<<<blocks, threads>>>(data, thrust::raw_pointer_cast(xHelp.data()));
    rearrangeY<<<blocks, threads>>>(data, thrust::raw_pointer_cast(yHelp.data()));

    thrust::device_vector<cPoint> points(data.ptsSize);

    arraysToPoints<<<blocks, threads>>>(data, thrust::raw_pointer_cast(points.data()));

    // sort upper part
    thrust::sort(thrust::device,
                 points.begin(),
                 points.begin() + splitIndex,
                 sortPointsAscending());

    // // sort lower part
    thrust::sort(thrust::device,
                 points.begin() + splitIndex,
                 points.end(),
                 sortPointsDescending());

    updateIndexes<<<blocks, threads>>>(data, thrust::raw_pointer_cast(points.data()));

    updateKeys<<<blocks, threads>>>(data, splitIndex);
}

#define BLOCK_SIZE 256

void CFirstSplit::firstSplit(qhData &data)
{
    flagsHelp.resize(data.ptsSize);
    xHelp.resize(data.ptsSize);
    yHelp.resize(data.ptsSize);

    blocks = ceil((double)data.ptsSize / BLOCK_SIZE);
    threads = BLOCK_SIZE;

    unsigned splitIndex = partitionPoints(data);

    sortPartitions(data, splitIndex);
}
