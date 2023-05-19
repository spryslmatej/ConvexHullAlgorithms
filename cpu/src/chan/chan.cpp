#include "chan.h"

void CChan::createSegments(vector<Segment> &segments,
                           const unsigned pointCount)
{
    unsigned step = pointCount / (gridDim - 1);

    unsigned i = 0;
    for (i = 0; i + step < pointCount; i += step)
    {
        segments.emplace_back(i, i + step);
    }

    if (i < pointCount)
        segments.emplace_back(i, pointCount);
}

void CChan::solve(vector<Point> &points, vector<Point> &hull)
{
    // create segments of points
    vector<Segment> segments;
    if (gridDim == 1)
        segments.emplace_back(0, points.size());
    else
    {
        if (gridDim > points.size())
            gridDim = points.size();

        createSegments(segments, points.size());
    }

    // fill hulls
    vector<vector<Point>> hulls(gridDim * gridDim, vector<Point>());

    assert(segments.size() == segments.size());

    // do parallel graham scans
    cout << "Grahams on " << gridDim * gridDim << " threads." << endl;
    CGrahamScan gs;
#pragma omp parallel for shared(points)
    for (unsigned i = 0; i < segments.size(); i++)
    {
        gs.solveOnSegment(points, segments.at(i), hulls.at(i));
    }

    // jarvis march
    jarvisMarchCrossProduct(hulls, hull);
}

#include "../jarvis/jarvis.h"

void CChan::jarvisMarchCrossProduct(const vector<vector<Point>> &hulls,
                                    vector<Point> &hull)
{
    // flatten
    vector<Point> flattenedPoints;

    for (auto h : hulls)
        for (auto item : h)
            flattenedPoints.push_back(item);

    // solve
    CJarvisMarch jm;
    jm.solve(flattenedPoints, hull);
}
