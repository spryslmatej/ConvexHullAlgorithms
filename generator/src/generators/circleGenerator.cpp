#include "circleGenerator.h"

void CCircleGenerator::generate(vector<Point> &points,
                                const unsigned count,
                                const unsigned maxVal)
{
    //  allocate
    points.reserve(count);

    int upper = ceil(maxVal / 2);
    Point center = Point(upper, upper);

    const double PI = 3.14159;
    double radius = maxVal / 2;
    double step = 2 * PI / count, angle = 0;

    // fill
    for (unsigned i = 0; i < count; i++, angle += step)
    {
        points.emplace_back(Point(center.X + radius * cos(angle),
                                  center.Y + radius * sin(angle)));
    }
}
