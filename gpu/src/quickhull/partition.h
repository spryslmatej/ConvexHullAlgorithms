#pragma once

#include "qhData.h"

struct partitionNotRemoved
{
    __device__ bool operator()(const FLAG_TYPE x);
};

struct partitionByFlag
{
    __device__ bool operator()(const FLAG_TYPE x);
};
