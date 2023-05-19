#include "crawl.h"

#include <iostream>
#include <cmath>

#include "../model/directions.h"

#include "../utilities/helpers.h"
#include "../utilities/partition.h"

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

using namespace std;

__global__ void crawlKernel(
    Segment *rawSegments,
    const unsigned gridDim,
    const int rowStep, const int colStep)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= gridDim * gridDim)
        return;

    int row = thIndex / gridDim;
    int col = thIndex % gridDim;
    if (row != 0 &&
        row != gridDim - 1 &&
        col != 0 &&
        col != gridDim - 1)
        return;

    while (1)
    {
        if (row < 0 ||
            col < 0 ||
            row > gridDim - 1 ||
            col > gridDim - 1)
            return;

        if (!rawSegments[row * gridDim + col].empty)
        {
            rawSegments[row * gridDim + col].viable = true;
            return;
        }

        row += rowStep;
        col += colStep;
    }
}

__global__ void markPointsRemoved(Point *points, unsigned n,
                                  unsigned *positions,
                                  Segment *segments)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= n)
        return;

    unsigned ptPos = positions[thIndex];

    if (segments[ptPos].viable == 0)
    {
        points[thIndex].removed = true;
    }
}

void CCrawl::copyToDevice()
{
    devSegments = hostSegments;
    devPositions = hostPositions;
}

void CCrawl::findExtremeValues(const thrust::host_vector<Point> &points)
{
    stats.xMin = points[0].X;
    stats.xMax = points[0].X;
    stats.yMin = points[0].Y;
    stats.yMax = points[0].Y;

    for (unsigned i = 0; i < points.size(); i++)
    {
        auto item = points[i];
        if (item.X < stats.xMin)
            stats.xMin = item.X;
        else if (item.X > stats.xMax)
            stats.xMax = item.X;
        if (item.Y < stats.yMin)
            stats.yMin = item.Y;
        else if (item.Y > stats.yMax)
            stats.yMax = item.Y;
    }
}

void CCrawl::createGrid(thrust::host_vector<Point> &points)
{
    findExtremeValues(points);

    hostSegments.reserve(gridDim * gridDim);

    unsigned step = (stats.xMax) > (stats.yMax)
                        ? (double)stats.xMax / gridDim
                        : (double)stats.yMax / gridDim;
    step++;

    for (unsigned i = 0; i < gridDim * gridDim; i++)
    {
        hostSegments.push_back(Segment());
    }

    for (unsigned i = 0; i < points.size(); i++)
    {
        unsigned row = (double)points[i].X / step;
        unsigned col = (double)points[i].Y / step;

        // cout << i << " r:" << row << " c:" << col << endl;
        unsigned segmentIndex = row * gridDim + col;

        if (segmentIndex >= hostSegments.size())
            printf("ERROR: createGrid out of bounds\n");

        hostSegments[segmentIndex].empty = false;
        hostPositions.push_back(segmentIndex);
    }

    copyToDevice();
}

#define BLOCK_SIZE 256

void CCrawl::crawl(thrust::host_vector<Point> &points)
{
    devPoints = points;

    // create grid (grid is oriented top-down, left-right)
    createGrid(points);

    // run for each direction
    for (int i = directions::U; i != directions::LU; i++)
    {
        int rs, cs;
        getStepsFromDirection((directions)i, rs, cs);

        crawlKernel<<<ceil((double)(gridDim * gridDim) / BLOCK_SIZE), BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(devSegments.data()),
            gridDim,
            rs, cs);
    }

    // remove points
    markPointsRemoved<<<ceil((double)(points.size()) / BLOCK_SIZE), BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(devPoints.data()),
        devPoints.size(),
        thrust::raw_pointer_cast(devPositions.data()),
        thrust::raw_pointer_cast(devSegments.data()));

    points = devPoints;
}
