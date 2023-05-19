#include "recursive.h"

#include "helpers.h"
#include "partition.h"

#include "../utilities/helpers.h"

#include <cmath>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/partition.h>

using namespace std;

unsigned __device__ getLowerPoint(qhData &data, unsigned ptIndex)
{
    return data.rawFirstPts[data.rawKeys[ptIndex]];
}

unsigned __device__ getUpperPoint(qhData &data, unsigned ptIndex)
{
    unsigned myKey = data.rawKeys[ptIndex];
    if (myKey + 1 < data.rawFirstPtsSize)
        return data.rawFirstPts[myKey + 1];
    else
    {
        return data.rawFirstPts[0];
    }
}

void __device__ removePoint(qhData &data, unsigned ptIndex)
{
    data.rawDistances[ptIndex] = 0;
    data.rawHead[ptIndex] = 0;
    data.rawKeys[ptIndex] = 0;
}

__device__ double getPointDistanceFromLine(
    const COORDS_TYPE p1x, const COORDS_TYPE p1y,
    const COORDS_TYPE p2x, const COORDS_TYPE p2y,
    const COORDS_TYPE p3x, const COORDS_TYPE p3y)
{
    return abs(((double)p2x - (double)p1x) * ((double)p1y - (double)p3y) -
               ((double)p1x - (double)p3x) * ((double)p2y - (double)p1y)) /
           sqrt(pow((double)p2x - (double)p1x, 2) +
                pow((double)p2y - (double)p1y, 2));
}

bool __device__ isPointInTriangle(const COORDS_TYPE x, const COORDS_TYPE y,
                                  const COORDS_TYPE p1x, const COORDS_TYPE p1y,
                                  const COORDS_TYPE p2x, const COORDS_TYPE p2y,
                                  const COORDS_TYPE p3x, const COORDS_TYPE p3y)
{
    double d1, d2, d3;
    bool has_neg, has_pos;

    d1 = signPosition(x, y, p1x, p1y, p2x, p2y);
    d2 = signPosition(x, y, p2x, p2y, p3x, p3y);
    d3 = signPosition(x, y, p3x, p3y, p1x, p1y);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

#pragma region findAndAddFarthestPoints

void __global__ calculateDistances(qhData data)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myPoint = data.rawIndexes[thIndex];

    if (data.rawHead[myPoint] == 1)
    {
        data.rawDistances[myPoint] = 0;
        return;
    }

    unsigned lowerPoint = getLowerPoint(data, myPoint),
             upperPoint = getUpperPoint(data, myPoint);

    if (lowerPoint == upperPoint ||
        myPoint == lowerPoint ||
        myPoint == upperPoint)
    {
        printf("ERROR: calculateDistances! %d on k %d: %d %d\n",
               myPoint, data.rawKeys[myPoint], lowerPoint, upperPoint);
        assert(0);
    }

    double val = getPointDistanceFromLine(
        data.rawX[lowerPoint], data.rawY[lowerPoint],
        data.rawX[upperPoint], data.rawY[upperPoint],
        data.rawX[myPoint], data.rawY[myPoint]);

    data.rawDistances[myPoint] = val;
}

void __global__ getAndAddMaxDistPoints(qhData data,
                                       KEYS_TYPE *reducedKeys,
                                       DISTANCES_TYPE *reducedValues,
                                       unsigned reducedSize,
                                       unsigned *indexes)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myPoint = data.rawIndexes[thIndex];

    // solves thread races in same distances
    if (data.rawHead[myPoint] == 1)
        return;

    unsigned myKey = data.rawKeys[myPoint];

    if (reducedValues[myKey] == data.rawDistances[myPoint] &&
        data.rawDistances[myPoint] != 0)
    {
        indexes[myKey] = myPoint;

        // printf("Adding to hull: k:%d i:%d\n", data.rawKeys[myPoint], myPoint);
        data.rawHead[myPoint] = 1;
    }
}

void CRecursive::findAndAddFarthestPoints(qhData &data,
                                          thrust::device_vector<unsigned> &maxDistIndexes)
{
    // data.transferAndPrint();
    calculateDistances<<<blocks, threads>>>(data);

    thrust::device_vector<KEYS_TYPE> reducedKeys(keysHelp.size());
    thrust::device_vector<DISTANCES_TYPE> reducedValues(distancesHelp.size());

    rearrangeKeys<<<blocks, threads>>>(data, thrust::raw_pointer_cast(keysHelp.data()));
    rearrangeDistances<<<blocks, threads>>>(data, thrust::raw_pointer_cast(distancesHelp.data()));

    // find the max distances in segments
    // keys and values need to be the same type!
    thrust::reduce_by_key(
        thrust::device,
        keysHelp.begin(),
        keysHelp.end(),
        distancesHelp.begin(),
        reducedKeys.begin(),
        reducedValues.begin(),
        thrust::equal_to<KEYS_TYPE>(),
        thrust::maximum<DISTANCES_TYPE>());

    if (reducedKeys.size() != reducedValues.size())
    {
        cout << "ERROR: Keys and values are not the same size!" << endl;
        throw "";
    }

    unsigned reducedSize = reducedKeys.size();

    maxDistIndexes.resize(reducedSize);
    getAndAddMaxDistPoints<<<blocks, threads>>>(
        data,
        thrust::raw_pointer_cast(reducedKeys.data()),
        thrust::raw_pointer_cast(reducedValues.data()),
        reducedSize,
        thrust::raw_pointer_cast(maxDistIndexes.data()));
}

#pragma endregion findAndAddFarthestPoints

#pragma region discardInteriorPoints

void __global__ kernelDiscardInteriorPoints(qhData data, unsigned *furthest)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myPoint = data.rawIndexes[thIndex];

    unsigned
        lowerPoint = getLowerPoint(data, myPoint),
        upperPoint = getUpperPoint(data, myPoint),
        furthestPoint = furthest[data.rawKeys[myPoint]];

    // points in hull and furthest points cannot be removed
    if (data.rawHead[myPoint] == 1 || myPoint == furthestPoint)
    {
        data.rawFlag[myPoint] = 1;
        return;
    }

    if (myPoint == lowerPoint ||
        myPoint == upperPoint ||
        myPoint == furthestPoint ||
        lowerPoint == upperPoint ||
        lowerPoint == furthestPoint ||
        upperPoint == furthestPoint)
    {
        printf("ERROR: discardInteriorPoints! %d on k: %d: %d %d %d\n",
               myPoint, data.rawKeys[myPoint], lowerPoint, upperPoint, furthestPoint);
        assert(0);
    }

    bool result = isPointInTriangle(
        data.rawX[myPoint], data.rawY[myPoint],
        data.rawX[lowerPoint], data.rawY[lowerPoint],
        data.rawX[upperPoint], data.rawY[upperPoint],
        data.rawX[furthestPoint], data.rawY[furthestPoint]);

    // 0 for invalid point, 1 for valid point
    if (result == 1)
    {
        data.rawFlag[myPoint] = 0;
        removePoint(data, myPoint);
    }
    else
        data.rawFlag[myPoint] = 1;
}

void CRecursive::discardInteriorPoints(qhData &data,
                                       thrust::device_vector<unsigned> &maxDistIndexes)
{
    // thrust::copy(maxDistIndexes.begin(), maxDistIndexes.end(),
    //              std::ostream_iterator<unsigned>(std::cout, " "));
    // cout << endl;

    kernelDiscardInteriorPoints<<<blocks, threads>>>(
        data, thrust::raw_pointer_cast(maxDistIndexes.data()));

    // prepare flags for partition
    rearrangeFlags<<<blocks, threads>>>(data, thrust::raw_pointer_cast(flagsHelp.data()));

    // partition points by flags
    assert(data.devIndexes.size() == flagsHelp.size());
    auto splittingPoint = thrust::stable_partition(
        thrust::device,
        data.devIndexes.begin(),
        data.devIndexes.end(),
        flagsHelp.begin(),
        partitionNotRemoved());

    unsigned splitIndex = thrust::distance(data.devIndexes.begin(), splittingPoint);

    // resize
    data.resize(splitIndex);

    keysHelp.resize(data.ptsSize);
    distancesHelp.resize(data.ptsSize);
    headsHelp.resize(data.ptsSize);
    flagsHelp.resize(data.ptsSize);
}

#pragma endregion discardInteriorPoints

#pragma region updateKeys

void __global__ kernelUpdateKeys(qhData data, KEYS_TYPE *updatedKeys)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myPointIndex = data.rawIndexes[thIndex];

    int keyToSet = updatedKeys[thIndex] - 1;

    // cannot check for rawFirstPtsSize, as it is not yet updated
    if (keyToSet < 0 || keyToSet >= data.ptsSize)
    {
        printf("ERROR: setting an incorrect key! %d: %d\n", myPointIndex, keyToSet);
        assert(0);
    }

    data.rawKeys[myPointIndex] = keyToSet;
}

void CRecursive::updateKeys(qhData &data)
{
    thrust::device_vector<KEYS_TYPE> updatedKeys(headsHelp.size());

    rearrangeHeads<<<blocks, threads>>>(data,
                                        thrust::raw_pointer_cast(headsHelp.data()));

    // prefix sum to get updated keys
    assert(updatedKeys.size() == headsHelp.size());
    thrust::inclusive_scan(thrust::device,
                           headsHelp.begin(),
                           headsHelp.end(),
                           updatedKeys.begin());

    // updatedKeys need to be decremented by 1 to represent indexes
    kernelUpdateKeys<<<blocks, threads>>>(data, thrust::raw_pointer_cast(updatedKeys.data()));
}

#pragma endregion updateKeys

#pragma region updateFirstPoints

void __global__ kernelUpdateFirstPoints(qhData data)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.ptsSize)
        return;

    unsigned myIndex = data.rawIndexes[thIndex];

    if (data.rawHead[myIndex] == 1)
    {
        data.rawFirstPts[data.rawKeys[myIndex]] = myIndex;
    }
}

void CRecursive::updateFirstPoints(qhData &data)
{
    kernelUpdateFirstPoints<<<blocks, threads>>>(data);

    rearrangeHeads<<<blocks, threads>>>(data, thrust::raw_pointer_cast(headsHelp.data()));

    int result = thrust::reduce(thrust::device,
                                headsHelp.begin(),
                                headsHelp.end(),
                                0,
                                thrust::plus<int>());

    data.rawFirstPtsSize = result;
}

#pragma endregion updateFirstPoints

#define BLOCK_SIZE 256

void CRecursive::recursive(qhData &data)
{
    keysHelp.resize(data.ptsSize);
    distancesHelp.resize(data.ptsSize);
    headsHelp.resize(data.ptsSize);
    flagsHelp.resize(data.ptsSize);

    blocks = ceil((double)data.ptsSize / BLOCK_SIZE);
    threads = BLOCK_SIZE;

    thrust::device_vector<unsigned> maxDistIndexes;

    findAndAddFarthestPoints(data, maxDistIndexes);

    discardInteriorPoints(data, maxDistIndexes);

    updateKeys(data);

    updateFirstPoints(data);

    // int num_gpus;
    // size_t free, total;
    // cudaGetDeviceCount(&num_gpus);
    // for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++)
    // {
    //     cudaSetDevice(gpu_id);
    //     int id;
    //     cudaGetDevice(&id);
    //     cudaMemGetInfo(&free, &total);
    //     cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;
    // }
}
