#include "quickhull.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "../model/point.h"
#include "../utilities/handleError.cu"
#include "../utilities/helpers.h"

#include "qhData.h"

#include "firstSplit.h"
#include "recursive.h"
#include "helpers.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/find.h>

#define BLOCK_SIZE 256

using namespace std;

bool CQuickHull::checkToContinue(qhData &data)
{
    thrust::device_vector<HEAD_TYPE> help(data.ptsSize);
    rearrangeHeads<<<ceil((double)data.ptsSize / BLOCK_SIZE), BLOCK_SIZE>>>(data, thrust::raw_pointer_cast(help.data()));

    int result = thrust::reduce(thrust::device, help.begin(), help.end());

    return result < data.ptsSize;
}

void CQuickHull::checkPreconditions()
{
    assert(sizeof(KEYS_TYPE) == sizeof(DISTANCES_TYPE));
}

void CQuickHull::solve(const Point *points, const unsigned n, Point **hull, unsigned *hullSize)
{
    if (n <= 3)
    {
        *hull = (Point *)malloc(n * sizeof(Point));
        for (unsigned i = 0; i < n; i++)
            *hull[i] = points[i];
        *hullSize = n;
        return;
    }

    checkPreconditions();

    // cout << "Hull started" << endl;

    qhData data;

    data.createStructure(points, n);

    cout << "Loaded points." << endl;

    data.copyToDevice();
    data.setRawPointers();

    // start of algorithm

    CFirstSplit firstSplit;
    firstSplit.firstSplit(data);

    cout << "Finished first split." << endl;

    // cout << "After first split:" << endl;
    // data.transferAndPrint();

    CRecursive recursive;

    unsigned iterations = 1;

    for (; 1; iterations++)
    {
        recursive.recursive(data);

        if (!checkToContinue(data))
        {
            cout << "Breaking recurse." << endl;
            break;
        }
    }
    // end of algorithm

    cout << "Finished algorithm in " << iterations << " iterations." << endl;
    // data.transferAndPrint();

    data.copyToHost();

    int result = thrust::reduce(thrust::device,
                                data.devHead.begin(),
                                data.devHead.end(),
                                0,
                                thrust::plus<int>());

    if (result != data.ptsSize)
    {
        cout << "ERROR: heads not set to true: " << result << " " << data.ptsSize << endl;
    }

    data.outputStructure(hull, hullSize);
    *hullSize = data.ptsSize;

    cout << "\nCheck errors after quickhull:" << endl;
    checkForErrors();
}
