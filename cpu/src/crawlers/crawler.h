#pragma once

#include <vector>

#include "../crawlers/crawlerSegment.h"

using namespace std;

class Crawler
{
    unsigned row, col;
    int rowStep, colStep;

    void step();

    bool checkOutOfBounds(const vector<vector<CrawlerSegment>> &segments) const;

public:
    Crawler(const unsigned r, const unsigned c,
            const int rs, const int cs);

    void crawl(vector<vector<CrawlerSegment>> &segments);
};
