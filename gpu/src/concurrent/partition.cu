#include "partition.h"

#include <thrust/partition.h>
#include <thrust/execution_policy.h>

unsigned partitionRemovedPoints(concData &data)
{
    auto splittingPoint =
        thrust::partition(thrust::device,
                          data.devPoints.begin(), data.devPoints.end(),
                          partitionRemoved());

    unsigned remainingPoints = thrust::distance(data.devPoints.begin(), splittingPoint);
    unsigned removedPoints = data.devPoints.size() - remainingPoints;

    data.resize(remainingPoints);
    return removedPoints;
}

unsigned partitionInHullPoints(concData &data)
{
    auto splittingPoint =
        thrust::partition(thrust::host,
                          data.hostPoints.begin(), data.hostPoints.end(),
                          partitionInHull());

    unsigned remainingPoints = thrust::distance(data.hostPoints.begin(), splittingPoint);
    unsigned removedPoints = data.hostPoints.size() - remainingPoints;

    data.resize(remainingPoints);
    return removedPoints;
}
