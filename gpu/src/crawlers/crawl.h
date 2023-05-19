#pragma once

#include "../model/point.h"
#include "../model/stats.h"
#include "../model/segment.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class CCrawl
{
    unsigned gridDim;

    Stats stats;

    thrust::device_vector<Point> devPoints;

    thrust::host_vector<Segment> hostSegments;
    thrust::device_vector<Segment> devSegments;

    thrust::host_vector<unsigned> hostPositions;
    thrust::device_vector<unsigned> devPositions;

    void createGrid(thrust::host_vector<Point> &points);

    void findExtremeValues(const thrust::host_vector<Point> &points);

    void copyToDevice();

public:
    CCrawl(const unsigned gd) : gridDim(gd) {}
    void crawl(thrust::host_vector<Point> &points);
};