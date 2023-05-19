#include "quickhull.h"

#include "helpers.h"

#include "omp.h"

void CQuickHull::findExtremaIndexes(vector<Point> &q, unsigned &a, unsigned &b)
{
    unsigned xMin = 0, xMax = 0;

    for (unsigned i = 0; i < q.size(); i++)
    {
        Point curPoint = q.at(i);

        if (curPoint.X < q.at(xMin).X)
            xMin = i;

        if (curPoint.X > q.at(xMax).X)
            xMax = i;
    }

    a = xMin;
    b = xMax;
}

void CQuickHull::placeExtremas(vector<Point> &q, unsigned &iX, unsigned &iY)
{
    if (iX != 0)
    {
        swapItems(q, 0, iX);
        if (iY == 0)
            iY = iX;
    }
    iX = 0;

    if (iY != 1)
    {
        swapItems(q, 1, iY);
    }
    iY = 1;
}

void CQuickHull::sortBySide(vector<Point> &q,
                            const Segment s,
                            const Point x,
                            const Point y, unsigned &splitter)
{
    unsigned i = s.begin, j = s.end;
    double val;

    for (; i < j;)
    {
        val = signPosition(q.at(i), x, y);

        if (val < 0)
        {
            j--;
            swapItems(q, i, j);
        }
        else
            i++;
    }

    splitter = i;
}

unsigned CQuickHull::getIndexOfFarthestPointFromLine(const vector<Point> &q,
                                                     const Line line,
                                                     const Segment s)
{
    double maxDist = 0;
    unsigned index = 0;

    for (unsigned i = s.begin; i < s.end; i++)
    {
        Point curPoint = q.at(i);

        double dist = getPointDistanceFromLine(line, curPoint);
        if (dist > maxDist)
        {
            maxDist = dist;
            index = i;
        }
    }

    return index;
}

Segment CQuickHull::removePointsInTriangle(vector<Point> &q,
                                           const Segment s,
                                           const Triangle triangle)
{
    if (s.begin >= s.end)
        return s;

    unsigned i = s.begin, j = s.end;
    unsigned removedCount = 0;

    for (; i < j;)
    {
        Point &p = q.at(i);

        if (isPointInTriangle(p, triangle))
        {
            p.removed = true;
            removedCount++;
            j--;
            swapItems(q, i, j);
        }
        else
            i++;
    }

    // cout << "Removed " << removedCount << " points." << endl;

    return Segment(s.begin, i);
}

void CQuickHull::sortPointsToSidesOfLine(vector<Point> &q,
                                         const Segment s,
                                         const Line line,
                                         Segment &greater, Segment &lesser)
{
    unsigned i = s.begin, j = s.end;
    float val;
    for (; i < j;)
    {
        val = signPosition(q.at(i), line.x, line.y);
        if (val < 0)
        {
            j--;
            swapItems(q, i, j);
        }
        else
            i++;
    }

    greater = Segment(s.begin, i);
    lesser = Segment(i, s.end);
}

void CQuickHull::recurse(vector<Point> &q,
                         Segment s,
                         const Point x,
                         const Point y)
{
    if (s.size() < 1)
        return;

    // find furthest point
    unsigned iFurthest = getIndexOfFarthestPointFromLine(q, Line(x, y), s);
    // add to hull
    q.at(iFurthest).inHull = true;
    // cout << "\nAdded " << q.at(iFurthest) << "; Parents: " << x << "; " << y << endl;

    // last item in part
    if (s.size() == 1)
    {
        // cout << "Last. " << q.at(iFurthest) << endl;
        return;
    }

    // move furthest to begin
    swapItems(q, iFurthest, s.begin);
    iFurthest = s.begin;
    s.begin += 1;

    Point furthest = q.at(iFurthest);

    // remove nonviable points
    Segment viablePoints = removePointsInTriangle(q, s, Triangle(x, y, furthest));

    if (viablePoints.size() == 0)
        return;

    // cout << "Viable: " << viablePoints << endl;

    // sort points to sides
    Segment greater, lesser;
    Point intersect = getPointFootToLine(Line(x, y), furthest);
    sortPointsToSidesOfLine(q, viablePoints, Line(furthest, intersect), greater, lesser);

    // cout << "Split parts: " << greater << "; " << lesser << endl;

    // figure out which side is which (this changes below and above the connecting line)
    Segment atX, atY;
    if ((greater.size() > 0 &&
         arePointsOnSameSideOfLine(Line(furthest, intersect), q.at(greater.begin), x)) ||
        (lesser.size() > 0 &&
         !arePointsOnSameSideOfLine(Line(furthest, intersect), q.at(lesser.begin), x)))
    {
        atX = greater;
        atY = lesser;
    }
    else
    {
        atX = lesser;
        atY = greater;
    }

#pragma omp task shared(q)
    recurse(q, atX, x, furthest);

    recurse(q, atY, furthest, y);
}

void CQuickHull::solve(vector<Point> &q, vector<Point> &hull)
{
    // cout << "Hull started" << endl;

    Segment curS = Segment(0, q.size());

    unsigned a, b;

    findExtremaIndexes(q, a, b);
    q.at(a).inHull = true;
    q.at(b).inHull = true;

    // cout << "Extremas: " << a << ": " << q.at(a) << " " << b << ": " << q.at(b) << endl;

    //  place extremas at beginning
    placeExtremas(q, a, b);
    curS.begin += 2;

    unsigned splitter;
    sortBySide(q, curS, q.at(a), q.at(b), splitter);

    // Points are now arranged: [a, b | {s>0} | {s<0} ]

    Segment greater = Segment(b + 1, splitter),
            lesser = Segment(splitter, q.size());

    // cout << "Segments: " << greater << "; " << lesser << endl;

#pragma omp parallel
#pragma omp single
    {
#pragma omp taskgroup // waits for child tasks!
        {
#pragma omp task shared(q)
            recurse(q, greater, q.at(a), q.at(b));

            recurse(q, lesser, q.at(a), q.at(b));
        }
    }

    for (auto it : q)
    {
        if (it.inHull)
            hull.push_back(it);
    }
}
