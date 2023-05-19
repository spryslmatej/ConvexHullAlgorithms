#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "point.cpp"

#include "generator.h"
#include "generators/clusterGenerator.h"
#include "generators/randomClusterGenerator.h"
#include "generators/circleGenerator.h"
#include "generators/fuzzyCircleGenerator.h"

using namespace std;

void generate(CGenerator &gen, vector<Point> &points,
              unsigned pointCount, unsigned maxValue)
{
    gen.generate(points, pointCount, maxValue);
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout << "Usage: <Point count> <Mode [0-3]}> <Max value> <Path>"
             << "\nModes:\n"
             << "\tCluster: 0"
             << "\tCluster random: 1"
             << "\tCircle: 2"
             << "\tCircle fuzzy: 3"
             << endl;
        return 0;
    }

    unsigned pointCount = atoi(argv[1]),
             mode = atoi(argv[2]),
             maxValue = atoi(argv[3]);
    string exportPath = string(argv[4]);

    if (mode < 0 || mode > 3)
    {
        cout << "Incorrect mode" << endl;
        return 0;
    }

    vector<Point> points = {};

    CClusterGenerator cluster;
    CRandomClusterGenerator rCluster;
    CCircleGenerator circle;
    CFuzzyCircleGenerator fCircle;

    switch (mode)
    {
    case 0:
        generate(cluster, points, pointCount, maxValue);
        break;
    case 1:
        generate(rCluster, points, pointCount, maxValue);
        break;
    case 2:
        generate(circle, points, pointCount, maxValue);
        break;
    case 3:
        generate(fCircle, points, pointCount, maxValue);
        break;
    }

    ofstream stream(exportPath);

    stream << "2\n"
           << to_string(pointCount) << "\n";

    for (auto item : points)
    {
        stream << item.X << " " << item.Y << "\n";
    }
    stream << flush;

    stream.close();
}
