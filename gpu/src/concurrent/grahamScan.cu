#include "grahamScan.h"

#include "../model/turn.h"

#include "../utilities/helpers.h"

#include "partition.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

#include <iostream>
#include <cmath>

using namespace std;

struct DualStack
{
    unsigned *data, topIndex;

    __device__ DualStack(unsigned *d, unsigned s)
    {
        data = d;
        topIndex = 0;
    }

    __device__ unsigned top() { return data[topIndex]; }
    __device__ unsigned nextToTop() { return data[topIndex - 1]; }
    __device__ unsigned size() { return topIndex; }
    __device__ void push(unsigned val)
    {
        topIndex++;
        data[topIndex] = val;
    }
    __device__ unsigned pop()
    {
        topIndex--;
        return data[topIndex + 1];
    }
};

__device__ void runThrough(concData &data, unsigned *stacks,
                           const unsigned start, const unsigned end)
{
    unsigned size = end - start;
    if (size <= 3)
    {
        return;
    }

    DualStack ds(stacks + start, size);

    ds.push(start);
    ds.push(start + 1);

    for (unsigned i = start + 2; i < end; i++)
    {
        while (ds.size() >= 2 &&
               crossProduct(data.rawPoints[ds.nextToTop()],
                            data.rawPoints[ds.top()],
                            data.rawPoints[i]) !=
                   Turn::CLW)
        {
            data.rawPoints[ds.top()].removed = true;
            ds.pop();
        }
        ds.push(i);
    }
}

__global__ void grahamScanKernel(concData data,
                                 unsigned *reducedCount, unsigned reducedSize,
                                 unsigned *stacks)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= reducedSize)
        return;

    unsigned start = reducedCount[thIndex];
    unsigned end = (thIndex == reducedSize - 1)
                       ? data.rawPointsSize
                       : reducedCount[thIndex + 1];

    runThrough(data, stacks, start, end);
}

__global__ void calculatePositions(concData data, unsigned *positions)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.rawPointsSize)
        return;

    unsigned row = (double)data.rawPoints[thIndex].X / data.step;
    unsigned col = (double)data.rawPoints[thIndex].Y / data.step;

    unsigned segmentIndex = row * data.gridDim + col;

    positions[thIndex] = segmentIndex;
}

__device__ void insertion_sort_kernel(Point *points,
                                      unsigned start, unsigned end,
                                      float *angles)
{
    unsigned i = start + 1;
    while (i < end)
    {
        unsigned j = i;

        while (j > start && angles[j - 1] < angles[j])
        {
            // swap
            Point tmp = points[j];
            points[j] = points[j - 1];
            points[j - 1] = tmp;

            float ftmp = angles[j];
            angles[j] = angles[j - 1];
            angles[j - 1] = ftmp;

            j--;
        }
        i++;
    }
}

__global__ void sortPointsInSegments(concData data, float *polarAngles,
                                     unsigned *reducedCounts, unsigned reducedSize)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= reducedSize)
        return;

    unsigned start = reducedCounts[thIndex];
    unsigned end = (thIndex == reducedSize - 1)
                       ? data.rawPointsSize
                       : reducedCounts[thIndex + 1];

    insertion_sort_kernel(data.rawPoints, start, end, polarAngles);
}

__global__ void calculatePolarAngles(concData data,
                                     float *angles,
                                     unsigned *positions,
                                     unsigned *segmentBeginnings)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.rawPointsSize)
        return;

    Point myPoint = data.rawPoints[thIndex];

    unsigned mySegment = positions[thIndex];

    if (mySegment >= data.gridDim * data.gridDim)
        printf("ERROR: Segment out of bounds\n");

    unsigned mySegmentBeginning = segmentBeginnings[mySegment];

    if (mySegmentBeginning >= data.rawPointsSize)
        printf("ERROR: Segment beginning out of bounds\n");

    Point lowestPoint = data.rawPoints[mySegmentBeginning];

    angles[thIndex] = atan2f(myPoint.Y - lowestPoint.Y, myPoint.X - lowestPoint.X);
}

__global__ void fillSegmentBeginnings(
    unsigned *reducedKeys,
    unsigned *reducedCounts, unsigned reducedSize,
    unsigned *segmentBeginnings)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= reducedSize)
        return;

    unsigned myKey = reducedKeys[thIndex];
    unsigned myCount = reducedCounts[thIndex];

    segmentBeginnings[myKey] = myCount;
}

#define BLOCK_SIZE 256

void CGrahamScan::grahamScan(concData &data)
{
    // sort points by their y-coordinate
    thrust::sort(thrust::device,
                 data.devPoints.begin(),
                 data.devPoints.end(),
                 sortPointsByYDescending());

    // calculate segment indexes of points
    thrust::device_vector<unsigned> positions(data.rawPointsSize);
    unsigned *rawPositions = thrust::raw_pointer_cast(positions.data());
    calculatePositions<<<ceil((double)data.rawPointsSize / BLOCK_SIZE), BLOCK_SIZE>>>(data, rawPositions);

    // sort points and positions according to calculated positions
    thrust::stable_sort_by_key(thrust::device,
                               positions.begin(),
                               positions.end(),
                               data.devPoints.begin(),
                               thrust::less<unsigned>());

    // reduce positions to get counts in segment
    thrust::device_vector<unsigned>
        reducedKeys(data.rawPointsSize),
        reducedCounts(data.rawPointsSize);

    // reduce positions to the present values and their counts
    thrust::pair<unsigned *, unsigned *> new_end = thrust::reduce_by_key(
        thrust::device,
        rawPositions,
        rawPositions + positions.size(),
        thrust::make_constant_iterator(1),
        thrust::raw_pointer_cast(reducedKeys.data()),
        thrust::raw_pointer_cast(reducedCounts.data()),
        thrust::equal_to<unsigned>(),
        thrust::plus<unsigned>());

    unsigned reducedSize = thrust::distance(
        thrust::raw_pointer_cast(reducedKeys.data()),
        new_end.first);

    // calculate prefix sum to get starting positions (inplace)
    thrust::exclusive_scan(thrust::device,
                           reducedCounts.begin(),
                           reducedCounts.end(),
                           reducedCounts.data());

    // thrust::copy(reducedKeys.begin(), reducedKeys.begin() + reducedSize, std::ostream_iterator<unsigned>(std::cout, "\t"));
    // cout << endl;
    // thrust::copy(reducedCounts.begin(), reducedCounts.begin() + reducedSize, std::ostream_iterator<unsigned>(std::cout, "\t"));
    // cout << endl;
    // thrust::copy(positions.begin(), positions.begin() + positions.size(),
    //              std::ostream_iterator<unsigned>(std::cout, "\t"));
    // cout << endl;

    thrust::device_vector<unsigned> segmentBeginnings(data.gridDim * data.gridDim);
    thrust::fill(segmentBeginnings.begin(), segmentBeginnings.end(), 0);

    // write the segment beginnings from reduced keys
    fillSegmentBeginnings<<<ceil((double)reducedSize / BLOCK_SIZE), BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(reducedKeys.data()),
        thrust::raw_pointer_cast(reducedCounts.data()),
        reducedSize,
        thrust::raw_pointer_cast(segmentBeginnings.data()));

    // calculate polar angles of points
    thrust::device_vector<float> polarAngles(data.rawPointsSize);
    calculatePolarAngles<<<ceil((double)data.rawPointsSize / BLOCK_SIZE), BLOCK_SIZE>>>(
        data,
        thrust::raw_pointer_cast(polarAngles.data()),
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(segmentBeginnings.data()));

    // sort points in segments
    sortPointsInSegments<<<ceil((double)reducedSize / BLOCK_SIZE), BLOCK_SIZE>>>(
        data,
        thrust::raw_pointer_cast(polarAngles.data()),
        thrust::raw_pointer_cast(reducedCounts.data()),
        reducedSize);

    // run graham scans provided with stacks
    thrust::device_vector<unsigned> stacks(data.rawPointsSize);
    grahamScanKernel<<<ceil((double)reducedSize / BLOCK_SIZE), BLOCK_SIZE>>>(
        data,
        thrust::raw_pointer_cast(reducedCounts.data()),
        reducedSize,
        thrust::raw_pointer_cast(stacks.data()));

    unsigned removedPoints = partitionRemovedPoints(data);
    cout << "Graham removed " << removedPoints << " points." << endl;
}