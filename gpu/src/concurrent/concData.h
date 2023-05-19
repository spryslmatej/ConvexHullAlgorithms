#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../model/point.h"
#include "../model/stats.h"
#include "../model/segment.h"

struct concData
{
    thrust::host_vector<Point> hostPoints;
    thrust::device_vector<Point> devPoints;

    thrust::host_vector<unsigned> hostPositions;
    thrust::device_vector<unsigned> devPositions;

    thrust::host_vector<Segment> hostSegments;
    thrust::device_vector<Segment> devSegments;

    Stats stats;

    Point *rawPoints;
    unsigned *rawPositions;
    Segment *rawSegments;
    unsigned rawPointsSize;
    unsigned rawSegmentsSize;

    unsigned gridDim;
    unsigned step;

    void reserve(unsigned pointsCount, unsigned dimensions);

    void resize(unsigned size);
    
    void copyToDevice();

    void copyToHost();

    void setRawPointers();

    // print from host vectors
    void printPoints();
    void printSegmentsEmpty();
    void printSegmentsViable();
    void printPositions();
};
