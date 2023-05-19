#include "clusterGenerator.h"

void CClusterGenerator::generate(vector<Point> &points,
                                 const unsigned count,
                                 const unsigned maxVal)
{
    //  allocate
    points.reserve(count);

    //  seed
    std::srand(0);

    // fill
    for (unsigned i = 0; i < count; i++)
        points.emplace_back(Point(std::rand() % maxVal, std::rand() % maxVal));
}
