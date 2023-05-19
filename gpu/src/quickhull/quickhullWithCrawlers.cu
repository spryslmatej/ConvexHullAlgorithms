#include "quickhullWithCrawlers.h"

#include "../crawlers/crawl.h"
#include "quickhull.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

void CQuickHullWithCrawlers::solve(const Point *points, const unsigned n,
                                   Point **hull, unsigned *hullSize)
{
    thrust::host_vector<Point> pointsCopy(n);
    thrust::copy(points, points + n, pointsCopy.begin());

    CCrawl ccrawl(gridDim);
    ccrawl.crawl(pointsCopy);

    auto splitter = thrust::partition(pointsCopy.begin(), pointsCopy.end(),
                                   [](const Point &p)
                                   { return p.removed == false; });

    unsigned splitIndex = thrust::distance(pointsCopy.begin(), splitter);

    cout << "Crawl removed " << pointsCopy.size() - splitIndex << " points." << endl;

    pointsCopy.resize(splitIndex);

    CQuickHull qh;

    qh.solve(thrust::raw_pointer_cast(pointsCopy.data()),
             pointsCopy.size(), hull, hullSize);
}
