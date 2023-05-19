#include "concurrent.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utilities/helpers.h"
#include "../model/directions.h"

#include "concData.h"
#include "crawl.h"
#include "grahamScan.h"
#include "jarvisMarch.h"

#include <iostream>

using namespace std;

void CConcurrentHull::findExtremeValues(const Point *points, const unsigned n,
                                        Stats &stats)
{
    stats.xMin = points[0].X;
    stats.xMax = points[0].X;
    stats.yMin = points[0].Y;
    stats.yMax = points[0].Y;

    for (unsigned i = 0; i < n; i++)
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

void CConcurrentHull::createGrid(const Point *points, const unsigned n,
                                 concData &data)
{
    findExtremeValues(points, n, data.stats);

    data.reserve(n, gridDim);

    unsigned step = (data.stats.xMax) > (data.stats.yMax)
                        ? (double)data.stats.xMax / gridDim
                        : (double)data.stats.yMax / gridDim;
    step++;

    data.step = step;

    for (unsigned i = 0; i < gridDim * gridDim; i++)
    {
        data.hostSegments.push_back(Segment());
    }

    for (unsigned i = 0; i < n; i++)
    {
        unsigned row = (double)points[i].X / step;
        unsigned col = (double)points[i].Y / step;

        // cout << i << " r:" << row << " c:" << col << endl;
        unsigned segmentIndex = row * gridDim + col;

        if (segmentIndex >= data.hostSegments.size())
            printf("ERROR: createGrid out of bounds\n");

        data.hostPoints.push_back(points[i]);
        data.hostPositions.push_back(segmentIndex);
        data.hostSegments[segmentIndex].empty = false;
    }

    data.copyToDevice();
}

void CConcurrentHull::outputHull(concData &data, Point **hull, unsigned *hullSize)
{
    *hullSize = data.hostPoints.size();
    cudaMallocHost((void **)(hull), *hullSize);

    Point *ptsRaw = thrust::raw_pointer_cast(data.hostPoints.data());

    thrust::copy(ptsRaw, ptsRaw + data.hostPoints.size(), *hull);
}

void CConcurrentHull::solve(const Point *points, const unsigned n,
                            Point **hull, unsigned *hullSize)
{
    if (n <= 3)
    {
        *hull = (Point *)malloc(n * sizeof(Point));
        for (unsigned i = 0; i < n; i++)
            *hull[i] = points[i];
        *hullSize = n;
        return;
    }

    concData data;

    // create grid (grid is oriented top-down, left-right)
    createGrid(points, n, data);

    data.setRawPointers();

    data.copyToDevice();

    // crawl
    CCrawl crawl;
    crawl.crawl(data);

    // graham scan -- all points not in hulls will be marked as removed
    CGrahamScan graham;
    graham.grahamScan(data);

    // jarvis march -- all points in hull will be marked as inHull
    CJarvisMarch jarvis;
    jarvis.jarvisMarch(data);

    // output
    outputHull(data, hull, hullSize);

    // data.printPoints();
}