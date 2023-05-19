#pragma once

#include <vector>

using namespace std;

struct CrawlerSegment
{
    unsigned row, col;
    vector<unsigned> pointIndexes;
    bool viable = false;

    CrawlerSegment() ;
    CrawlerSegment(const unsigned r, const unsigned c);
};
