#pragma once

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../model/point.h"

#define INDEXES_TYPE int
#define HEAD_TYPE int
#define FLAG_TYPE int
#define FIRSTPTS_TYPE int
#define KEYS_TYPE int

#define COORDS_TYPE int

#define DISTANCES_TYPE float

using namespace std;

// indexes is used for sorting
// access all arrays by x[indexes[i]]! - they are merely a databank
struct qhData
{
    thrust::host_vector<INDEXES_TYPE> hostIndexes;
    thrust::host_vector<HEAD_TYPE> hostHead;
    thrust::host_vector<FLAG_TYPE> hostFlag;
    thrust::host_vector<FIRSTPTS_TYPE> hostFirstPts;
    thrust::host_vector<COORDS_TYPE> hostX;
    thrust::host_vector<COORDS_TYPE> hostY;
    thrust::host_vector<DISTANCES_TYPE> hostDistances;
    thrust::host_vector<KEYS_TYPE> hostKeys;

    thrust::device_vector<INDEXES_TYPE> devIndexes;
    thrust::device_vector<HEAD_TYPE> devHead;
    thrust::device_vector<FLAG_TYPE> devFlag;
    thrust::device_vector<FIRSTPTS_TYPE> devFirstPts;
    thrust::device_vector<COORDS_TYPE> devX;
    thrust::device_vector<COORDS_TYPE> devY;
    thrust::device_vector<DISTANCES_TYPE> devDistances;
    thrust::device_vector<KEYS_TYPE> devKeys;

    unsigned ptsSize = 0, rawFirstPtsSize = 0;

    INDEXES_TYPE *rawIndexes;
    HEAD_TYPE *rawHead;
    FLAG_TYPE *rawFlag;
    FIRSTPTS_TYPE *rawFirstPts;
    COORDS_TYPE *rawX, *rawY;
    DISTANCES_TYPE *rawDistances;
    KEYS_TYPE *rawKeys;

    void resize(unsigned n);

    void setRawPointers();

    // INPUT OUTPUT

    void createStructure(const Point *points, const unsigned n);

    void outputStructure(Point **hull, unsigned *hullSize) const;

    // COPY

    void copyToDevice();

    void copyToHost();

    // PRINT

    void transferAndPrint();

    void printHostData() const;
};
