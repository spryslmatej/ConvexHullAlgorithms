#include "jarvisMarch.h"

#include <thrust/extrema.h>
#include <thrust/distance.h>

#include "partition.h"

#include "../model/turn.h"

#include "../utilities/helpers.h"

using namespace std;

__global__ void addAllPoints(concData data)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.rawPointsSize)
        return;

    data.rawPoints[thIndex].inHull = true;
}


void CJarvisMarch::jarvisMarch(concData &data)
{
    // if point count is less or equal to 3, add all points to hull
    if (data.rawPointsSize <= 3)
    {
        addAllPoints<<<1, data.rawPointsSize>>>(data);
        return;
    }

    auto smallest = thrust::min_element(
        thrust::device,
        data.rawPoints,
        data.rawPoints + data.rawPointsSize,
        sortPointsByXAscending());

    unsigned min = thrust::distance(data.rawPoints, smallest);

    // cout << "Smallest x index: " << min << endl;

    unsigned curIndex = min, nextIndex;

    // continue the jarvis march on host as it will be faster
    data.copyToHost();

    do
    {
        data.hostPoints[curIndex].inHull = true;

        nextIndex = (curIndex + 1) % data.hostPoints.size();
        for (unsigned i = 0; i < data.hostPoints.size(); i++)
        {
            if (crossProduct(data.hostPoints[curIndex],
                             data.hostPoints[i],
                             data.hostPoints[nextIndex]) == Turn::CCW)
                nextIndex = i;
        }
        curIndex = nextIndex;
    } while (curIndex != min);

    unsigned removedPoints = partitionInHullPoints(data);
    cout << "Jarvis removed " << removedPoints << " points." << endl;
}
