#include "crawlers.h"

#include "omp.h"

void CCrawlers::createGrid(const vector<Point> &points,
                           const unsigned gridDim,
                           vector<vector<CrawlerSegment>> &segments,
                           Stats &stats,
                           unsigned &step)
{
    findExtremeValues(points, stats);

    step = (stats.xMax - stats.xMin) > (stats.yMax - stats.yMin)
               ? (stats.xMax - stats.xMin) / gridDim
               : (stats.yMax - stats.yMin) / gridDim;

    // initialize segments
    vector<CrawlerSegment> seg(gridDim + 1);
    segments = vector<vector<CrawlerSegment>>(gridDim + 1, seg);

#pragma omp parallel for
    for (unsigned i = 0; i <= gridDim; i++)
    {
        for (unsigned j = 0; j <= gridDim; j++)
        {
            segments.at(i).at(j).row = i;
            segments.at(i).at(j).row = j;
        }
    }
}
void CCrawlers::crawl(vector<vector<CrawlerSegment>> &segments,
                      const unsigned i,
                      const unsigned j)
{
    vector<Crawler> crawlers;
    CrawlerFactory::createCrawlersFromPoint(segments, i, j, crawlers);

#pragma omp parallel for
    for (auto it : crawlers)
        it.crawl(segments);
}

void CCrawlers::runCrawlers(const unsigned gridDim,
                            vector<vector<CrawlerSegment>> &segments)
{
#pragma omp parallel for
    for (unsigned i = 0; i <= gridDim; i++)
    {
        for (unsigned j = 0; j <= gridDim; j++)
        {
            if (i == 0 ||
                j == 0 ||
                i == segments.size() - 1 ||
                j == segments.at(0).size() - 1)
                crawl(segments, i, j);
        }
    }
}

void CCrawlers::sortPointsIntoSegments(const vector<Point> &points,
                                       vector<vector<CrawlerSegment>> &segments,
                                       const Stats stats,
                                       const unsigned step)
{
    for (unsigned i = 0; i < points.size(); i++)
    {
        unsigned row = (points.at(i).X - stats.xMin) / step;
        unsigned col = (points.at(i).Y - stats.yMin) / step;

        segments.at(row).at(col).pointIndexes.push_back(i);
    }
}

void CCrawlers::flagNonviablePoints(vector<Point> &points,
                                    const vector<vector<CrawlerSegment>> &segments)
{
#pragma omp parallel for
    for (auto row : segments)
    {
        for (auto it : row)
        {
            if (!it.viable)
            {
                for (auto p : it.pointIndexes)
                {
                    points.at(p).removed = true;
                }
            }
        }
    }
}

void CCrawlers::crawlersQH(vector<Point> &points,
                           const unsigned gridDim)
{
    Stats stats;
    unsigned step;
    vector<vector<CrawlerSegment>> segments;
    createGrid(points, gridDim, segments, stats, step);

    // sort points into segments
    sortPointsIntoSegments(points, segments, stats, step);

    // run crawlers from each boundary segment
    runCrawlers(gridDim, segments);

    // remove points not in viable segments
    flagNonviablePoints(points, segments);
}

void CCrawlers::crawlersCO(const vector<Point> &points,
                           const unsigned gridDim,
                           vector<CrawlerSegment> &viableSegments)
{
    Stats stats;
    unsigned step;
    vector<vector<CrawlerSegment>> segments;

    CCrawlers crawlers;
    crawlers.createGrid(points, gridDim, segments, stats, step);

    // sort points into segments
    sortPointsIntoSegments(points, segments, stats, step);

    // run crawlers from each boundary segment
    runCrawlers(gridDim, segments);

    // remove points not in viable segments
    for (auto row : segments)
    {
        for (auto it : row)
        {
            if (it.viable)
                viableSegments.push_back(it);
        }
    }
}
