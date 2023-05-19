#include "qhData.h"

#include "../utilities/handleError.cu"

void qhData::resize(unsigned n)
{
    if (n == 0)
        cout << "WARNING: Resizing to 0" << endl;

    ptsSize = n;

    // do not resize other vectors - they are merely a databank
    hostIndexes.resize(n);
    hostFirstPts.resize(n);

    devIndexes.resize(n);
    devFirstPts.resize(n);
}

void qhData::setRawPointers()
{
    rawIndexes = thrust::raw_pointer_cast(devIndexes.data());
    rawHead = thrust::raw_pointer_cast(devHead.data());
    rawFlag = thrust::raw_pointer_cast(devFlag.data());
    rawFirstPts = thrust::raw_pointer_cast(devFirstPts.data());
    rawX = thrust::raw_pointer_cast(devX.data());
    rawY = thrust::raw_pointer_cast(devY.data());
    rawDistances = thrust::raw_pointer_cast(devDistances.data());
    rawKeys = thrust::raw_pointer_cast(devKeys.data());
}

void qhData::createStructure(const Point *points, const unsigned n)
{
    ptsSize = n;
    rawFirstPtsSize = 0;

    for (unsigned i = 0; i < ptsSize; i++)
    {
        hostIndexes.push_back(i);
        hostHead.push_back(0);
        hostFlag.push_back(0);
        hostKeys.push_back(0);
        hostDistances.push_back(0);
        hostX.push_back(points[i].X);
        hostY.push_back(points[i].Y);

        hostFirstPts.push_back(0);
    }
}

void qhData::outputStructure(Point **hull, unsigned *hullSize) const
{
    unsigned size = ptsSize;
    *hullSize = size;

    Point *h;
    cudaMallocHost((void **)&h, size * sizeof(Point));

    unsigned cur = 0;

    for (unsigned i = 0; cur < size; i++)
    {
        INDEXES_TYPE curIndex = hostIndexes[i];

        if (hostHead[curIndex] == 1)
        {
            h[cur] = Point(hostX[curIndex], hostY[curIndex]);
            cur++;
        }
    }

    *hull = h;
}

void qhData::copyToDevice()
{
    devIndexes = hostIndexes;
    devX = hostX;
    devY = hostY;
    devDistances = hostDistances;
    devHead = hostHead;
    devFlag = hostFlag;
    devKeys = hostKeys;
    devFirstPts = hostFirstPts;

    ptsSize = hostIndexes.size();
    rawFirstPtsSize = hostFirstPts.size();
}

void qhData::copyToHost()
{
    hostIndexes = devIndexes;
    hostX = devX;
    hostY = devY;
    hostDistances = devDistances;
    hostHead = devHead;
    hostFlag = devFlag;
    hostKeys = devKeys;
    hostFirstPts = devFirstPts;
}

void qhData::transferAndPrint()
{
    cudaDeviceSynchronize();
    copyToHost();
    printHostData();
}

void qhData::printHostData() const
{
    cout << "\nData:\n";
    for (unsigned i = 0; i < hostIndexes.size(); i++)
    {
        unsigned ptIndex = hostIndexes[i];
        cout << i << fixed
             << "\ti:" << hostIndexes[i]
             << "\tx:" << hostX[ptIndex]
             << "\ty:" << hostY[ptIndex]
             << "\td:" << hostDistances[ptIndex] << "\t"
             << "\th:" << hostHead[ptIndex]
             << "\tk:" << hostKeys[ptIndex]
             << "\tf:" << hostFlag[ptIndex] << "\n";
    }
    cout << "firstPts: [" << rawFirstPtsSize << "]\n";

    for (unsigned i = 0; i < rawFirstPtsSize; i++)
        cout << hostFirstPts[i] << " ";

    cout << "\n"
         << endl;
}
