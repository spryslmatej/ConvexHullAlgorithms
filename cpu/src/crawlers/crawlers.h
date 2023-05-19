#pragma once

#include <vector>
#include <iostream>
#include "crawler.h"
#include "crawlerFactory.cpp"
#include "../crawlers/crawlerSegment.h"

#include "../model/point.h"
#include "../utilities/values.h"

using namespace std;
class CCrawlers
{
    void createGrid(const vector<Point> &points,
                    const unsigned gridDim,
                    vector<vector<CrawlerSegment>> &segments,
                    Stats &stats,
                    unsigned &step);

    void crawl(vector<vector<CrawlerSegment>> &segments,
               const unsigned i,
               const unsigned j);

    void sortPointsIntoSegments(const vector<Point> &points,
                                vector<vector<CrawlerSegment>> &segments,
                                const Stats stats,
                                const unsigned step);

    void flagNonviablePoints(vector<Point> &points,
                             const vector<vector<CrawlerSegment>> &segments);

public:
    void runCrawlers(const unsigned gridDim,
                     vector<vector<CrawlerSegment>> &segments);

    void crawlersQH(vector<Point> &points,
                    const unsigned gridDim);

    void crawlersCO(const vector<Point> &points,
                    const unsigned gridDim,
                    vector<CrawlerSegment> &viableSegments);
};
