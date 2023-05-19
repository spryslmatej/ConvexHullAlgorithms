#include "quickhullWithCrawlers.h"

#include "../crawlers/crawlers.h"
#include "quickhull.h"

void CQuickHullWithCrawlers::solve(vector<Point> &q, vector<Point> &hull)
{
    CCrawlers crawlers;
    crawlers.crawlersQH(q, gridDim);

    auto splitter = std::partition(q.begin(), q.end(), [](const Point &p)
                                   { return p.removed == false; });

    unsigned splitIndex = std::distance(q.begin(), splitter);

    cout << "Crawl removed " << q.size() - splitIndex << " points." << endl;

    q.resize(splitIndex);

    CQuickHull qh;

    qh.solve(q, hull);
}
