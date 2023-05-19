#include "fuzzyCircleGenerator.h"

void CFuzzyCircleGenerator::generate(vector<Point> &points,
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

    //  fuzz
    std::srand(time(NULL));
    unsigned fuzzConstant = maxVal / 10;
    radius -= fuzzConstant;

    // fill
    for (unsigned i = 0; i < count; i++, angle += step)
    {
        int fuzzVal1 = std::rand() % fuzzConstant,
            fuzzVal2 = std::rand() % fuzzConstant;
        fuzzVal1 = std::rand() % 2 == 0 ? -fuzzVal1 : fuzzVal1;
        fuzzVal2 = std::rand() % 2 == 0 ? -fuzzVal2 : fuzzVal2;

        int x = center.X + radius * cos(angle) + fuzzVal1,
            y = center.Y + radius * sin(angle) + fuzzVal2;

        if (x < 0)
            x = 0;
        else if (x > (int)maxVal)
            x = maxVal;

        if (y < 0)
            y = 0;
        else if (y > (int)maxVal)
            y = maxVal;

        points.emplace_back(Point(x, y));
    }
}
