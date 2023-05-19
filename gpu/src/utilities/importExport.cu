#pragma once

#include "handleError.cu"

#include "../model/point.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <fstream>

using namespace std;

void importPoints(Point **p, unsigned *n, const string path)
{
    std::ifstream stream(path);

    unsigned dimension, count;
    stream >> dimension;
    stream.ignore(100, '\n');
    stream >> count;

    Point *points;
    HANDLE_ERROR(cudaHostAlloc((void **)&points, count * sizeof(Point), cudaHostAllocDefault));

    for (unsigned i = 0; i < count; i++)
    {
        Point curPoint;
        stream >> curPoint.X >> curPoint.Y;
        points[i] = curPoint;
    }

    stream.close();

    *p = points;
    *n = count;
}

void exportPoints(Point *p, unsigned n, const string path)
{
    // sort so the outputs can be compared
    thrust::sort(thrust::host,
                 p,
                 p + n,
                 [](const Point &x, const Point &y)
                 {
                     return (x.X > y.X) || (x.X == y.X && x.Y > y.Y);
                 });

    ofstream stream(path);

    for (unsigned i = 0; i < n; i++)
    {
        if (p[i].removed)
        {
            cout << "ERROR: Exporting removed point!" << endl;
            throw "ERROR: Exporting removed point!";
        }
        stream << p[i].X << " " << p[i].Y << "\n";
    }
    stream << flush;

    stream.close();
}
