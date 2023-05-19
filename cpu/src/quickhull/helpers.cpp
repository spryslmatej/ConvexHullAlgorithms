#include "helpers.h"

void swapItems(vector<Point> &q, const unsigned x, const unsigned y)
{
    Point tmp = q.at(x);
    q.at(x) = q.at(y);
    q.at(y) = tmp;
}

//  MATH

double signPosition(const Point p1, const Point p2, const Point p3)
{
    return (p1.X - p3.X) * (p2.Y - p3.Y) - (p2.X - p3.X) * (p1.Y - p3.Y);
}

double getPointDistanceFromLine(const Line l, const Point z)
{
    Point x = l.x,
          y = l.y;

    return abs((y.X - x.X) * (x.Y - z.Y) -
               (x.X - z.X) * (y.Y - x.Y)) /
           sqrt(pow(y.X - x.X, 2) + pow(y.Y - x.Y, 2));
}

Point getPointFootToLine(const Line l,
                         const Point p)
{
    double a, b, c;

    Point x = l.x,
          y = l.y;

    a = x.Y - y.Y;
    b = y.X - x.X;
    c = (x.X - y.X) * x.Y + (y.Y - x.Y) * x.X;

    double tmp = -1 * (a * p.X + b * p.Y + c) / (a * a + b * b);
    double resX = tmp * a + p.X;
    double resY = tmp * b + p.Y;
    return Point(resX, resY);
}

//  tests if both points p and q are on the same side of line
bool arePointsOnSameSideOfLine(const Line l,
                               const Point p, const Point q)
{
    float val1 = signPosition(l.x, l.y, p),
          val2 = signPosition(l.x, l.y, q);
    return ((val1 > 0 && val2 > 0) ||
            (val1 < 0 && val2 < 0) ||
            (val1 == 0 && val2 == 0));
}

bool isPointInTriangle(const Point pt, const Triangle t)
{
    Point v1 = t.x,
          v2 = t.y,
          v3 = t.z;

    double d1 = signPosition(pt, v1, v2),
           d2 = signPosition(pt, v2, v3),
           d3 = signPosition(pt, v3, v1);

    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0),
         has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}
