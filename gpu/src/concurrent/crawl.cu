#include "crawl.h"

#include <iostream>
#include <cmath>

#include "partition.h"

#include "../model/directions.h"

#include "../utilities/helpers.h"

using namespace std;

__global__ void crawlKernel(concData data,
                            const int rowStep, const int colStep)
{
    unsigned thIndex = getThIndex();

    unsigned gridDim = data.gridDim;

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

        if (!data.rawSegments[row * gridDim + col].empty)
        {
            data.rawSegments[row * gridDim + col].viable = true;
            return;
        }

        row += rowStep;
        col += colStep;
    }
}

__global__ void markPointsRemoved(concData data)
{
    unsigned thIndex = getThIndex();

    if (thIndex >= data.rawPointsSize)
        return;

    unsigned ptPos = data.rawPositions[thIndex];

    if (data.rawSegments[ptPos].viable == 0)
    {
        data.rawPoints[thIndex].removed = true;
    }
}


#define BLOCK_SIZE 256

void CCrawl::crawl(concData &data)
{
    // run for each direction
    for (int i = directions::U; i != directions::LU; i++)
    {
        int rs, cs;
        getStepsFromDirection((directions)i, rs, cs);

        crawlKernel<<<ceil((double)(data.gridDim * data.gridDim) / BLOCK_SIZE), BLOCK_SIZE>>>(
            data, rs, cs);
    }

    // remove points
    markPointsRemoved<<<ceil((double)(data.rawPointsSize) / BLOCK_SIZE), BLOCK_SIZE>>>(data);

    unsigned removedPoints = partitionRemovedPoints(data);
    cout << "Crawl removed " << removedPoints << " points." << endl;
}
