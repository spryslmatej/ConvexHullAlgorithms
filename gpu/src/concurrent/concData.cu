#include "concData.h"

#include <iostream>

void concData::reserve(unsigned pointsCount, unsigned dimensions)
{
    hostPoints.reserve(pointsCount);
    hostPositions.reserve(pointsCount);
    gridDim = dimensions;
    hostSegments.reserve(gridDim * gridDim);
}

void concData::resize(unsigned size)
{
    hostPoints.resize(size);
    devPoints.resize(size);
    rawPointsSize = size;
}

void concData::copyToDevice()
{
    devPoints = hostPoints;
    devPositions = hostPositions;
    devSegments = hostSegments;
}

void concData::copyToHost()
{
    hostPoints = devPoints;
    hostPositions = devPositions;
    hostSegments = devSegments;
}

void concData::setRawPointers()
{
    rawPoints = thrust::raw_pointer_cast(devPoints.data());
    rawPositions = thrust::raw_pointer_cast(devPositions.data());
    rawSegments = thrust::raw_pointer_cast(devSegments.data());
    rawPointsSize = devPoints.size();
    rawSegmentsSize = devSegments.size();
}

using namespace std;
void concData::printPoints()
{
    copyToHost();

    cout << "\nData:\n";
    for (unsigned i = 0; i < hostPoints.size(); i++)
    {
        cout << i
             << "\tx:" << hostPoints[i].X
             << "\ty:" << hostPoints[i].Y
             << "\th:" << hostPoints[i].inHull
             << "\tr:" << hostPoints[i].removed << "\n";
    }

    cout << endl;
}

void concData::printSegmentsEmpty()
{
    cout << "\nSegments - Empty:\n";
    for (unsigned i = 0; i < gridDim; i++)
    {
        for (unsigned j = 0; j < gridDim; j++)
            cout << hostSegments[i * gridDim + j].empty;
        cout << "\n";
    }
    cout << endl;
}

void concData::printSegmentsViable()
{
    cout << "\nSegments - Viable:\n";
    for (unsigned i = 0; i < gridDim; i++)
    {
        for (unsigned j = 0; j < gridDim; j++)
            cout << hostSegments[i * gridDim + j].viable;
        cout << "\n";
    }
    cout << endl;
}

void concData::printPositions()
{
    cout << "\nPositions:\n";
    for (unsigned i = 0; i < hostPoints.size(); i++)
    {
        cout << i << " " << hostPositions[i] << endl;
    }
    cout << endl;
}
