#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "../model/point.h"

using namespace std;

void importVector(vector<Point> &q, const string path)
{
    ifstream stream = ifstream(path);

    unsigned dimension, count;
    stream >> dimension;
    stream.ignore(100, '\n');
    stream >> count;

    q.reserve(count);

    for(unsigned i = 0; i < count; i++){
        Point curPoint;
        stream >> curPoint.X >> curPoint.Y;
        q.emplace_back(curPoint);
    }

    stream.close();
}
