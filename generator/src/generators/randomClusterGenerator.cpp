#include "randomClusterGenerator.h"

void CRandomClusterGenerator::generate(vector<Point> &points,
                                       const unsigned count,
                                       const unsigned maxVal)
{
    //  allocate
    points.reserve(count);

    //  seed
    std::srand(time(NULL));

    // fill
    for (unsigned i = 0; i < count; i++)
        points.emplace_back(Point(std::rand() % maxVal, std::rand() % maxVal));
}
