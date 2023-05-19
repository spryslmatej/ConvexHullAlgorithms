#pragma once

#include "qhData.h"

#include <thrust/device_vector.h>

class CRecursive
{
    thrust::device_vector<KEYS_TYPE> keysHelp;
    thrust::device_vector<DISTANCES_TYPE> distancesHelp;
    thrust::device_vector<HEAD_TYPE> headsHelp;
    thrust::device_vector<FLAG_TYPE> flagsHelp;

    unsigned blocks, threads;

    void findAndAddFarthestPoints(
        qhData &data,
        thrust::device_vector<unsigned> &maxDistIndexes);

    void discardInteriorPoints(
        qhData &data,
        thrust::device_vector<unsigned> &maxDistIndexes);

    void updateKeys(qhData &data);

    void updateFirstPoints(qhData &data);

public:
    void recursive(qhData &data);
};
