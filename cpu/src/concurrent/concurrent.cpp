#include "concurrent.h"

#include "../utilities/values.h"

#include "../chan/chan.h"

#include "../crawlers/crawler.h"
#include "../crawlers/crawlerFactory.cpp"
#include "../crawlers/crawlers.h"

void CConcurrentHull::solve(vector<Point> &points, vector<Point> &hull)
{
    vector<CrawlerSegment> segments;
    
    // crawl
    // get viable segments from crawl
    CCrawlers crawlers;
    crawlers.crawlersCO(points, gridDim, segments);

    vector<vector<Point>> linearSegments;

    for (auto it : segments)
    {
        vector<Point> segment;
        for (auto pi : it.pointIndexes)
            segment.push_back(points.at(pi));

        linearSegments.push_back(segment);
    }

    // input into chan
    // return result
    chanConc(linearSegments, hull);
}

void CConcurrentHull::chanConc(vector<vector<Point>> &segments, vector<Point> &hull)
{
    // fill hulls
    vector<vector<Point>> hulls(segments.size(), vector<Point>());

    // graham scans
    CGrahamScan gs;
#pragma omp parallel for
    for (unsigned i = 0; i < segments.size(); i++)
    {
        gs.solve(segments.at(i), hulls.at(i));
    }

    // jarvis march
    CChan ch(gridDim);
    ch.jarvisMarchCrossProduct(hulls, hull);
}
