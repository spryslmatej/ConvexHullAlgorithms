#pragma once

#include "qhData.h"

#include <thrust/device_vector.h>

class CFirstSplit
{
    thrust::device_vector<FLAG_TYPE> flagsHelp;
    thrust::device_vector<COORDS_TYPE> xHelp;
    thrust::device_vector<COORDS_TYPE> yHelp;

    unsigned blocks, threads;

    void findExtremePoints(const qhData data, unsigned &minIndex, unsigned &maxIndex);
    unsigned partitionPoints(qhData &data);
    void sortPartitions(qhData &data, unsigned splitIndex);

public:
    void firstSplit(qhData &data);
};
