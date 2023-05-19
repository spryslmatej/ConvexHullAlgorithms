#include "crawler.h"

Crawler::Crawler(const unsigned r, const unsigned c,
                 const int rs, const int cs)
    : row(r), col(c), rowStep(rs), colStep(cs) {}

void Crawler::step()
{
    row += rowStep;
    col += colStep;
}

bool Crawler::checkOutOfBounds(const vector<vector<CrawlerSegment>> &segments) const
{
    return (row < 0 ||
            col < 0 ||
            row > segments.size() - 1 ||
            col > segments.at(0).size() - 1);
}

void Crawler::crawl(vector<vector<CrawlerSegment>> &segments)
{
    CrawlerSegment curSeg = segments.at(row).at(col);
    while (true)
    {
        if (!curSeg.pointIndexes.empty())
            break;
        step();
        if (checkOutOfBounds(segments))
            return;
        curSeg = segments.at(row).at(col);
    }

    segments.at(row).at(col).viable = true;
}
