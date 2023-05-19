#include "point.h"

using namespace std;

Point::Point() {}
Point::Point(POINT_DATATYPE x, POINT_DATATYPE y)
    : X(x), Y(y), removed(false), inHull(false) {}
Point::Point(const Point &x)
    : X(x.X), Y(x.Y), removed(x.removed), inHull(x.inHull) {}

bool operator<(const Point l, const Point r)
{
    if (l.X < r.X)
        return true;
    else if (l.X == r.X)
        return l.Y < r.Y;
    else
        return false;
}

bool operator==(const Point l, const Point r)
{
    return l.X == r.X && l.Y == r.Y;
}

bool operator!=(const Point l, const Point r)
{
    return !(l == r);
}

std::ostream &operator<<(std::ostream &out, const Point &p)
{
    return out << p.X << ", " << p.Y << "; Hull: " << p.inHull << ", Removed: " << p.removed << "; ";
}
