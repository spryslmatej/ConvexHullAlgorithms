#pragma once

#include <iostream>
#include <fstream>

#include <vector>
#include <set>
#include <algorithm>

#include "../model/point.h"

using namespace std;

void exportVector(vector<Point> &q, const string path)
{
    std::sort(q.begin(), q.end(), [](Point x, Point y)
              { return x.X > y.X || (x.X == y.X && x.Y > y.Y); });

    ofstream stream = ofstream(path);

    for (auto item : q)
    {
        stream << item.X << " " << item.Y << "\n";
    }
    stream << flush;

    stream.close();
}

void exportSet(const set<Point> &q, const string path)
{
    ofstream stream = ofstream(path);

    for (auto item : q)
    {
        stream << item.X << " " << item.Y << "\n";
    }
    stream << flush;

    stream.close();
}
