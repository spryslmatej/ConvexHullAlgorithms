#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "concurrent/concurrent.h"
#include "quickhull/quickhull.h"
#include "quickhull/quickhullWithCrawlers.h"

#include "utilities/importExport.cu"
#include "utilities/handleError.cu"
#include "utilities/helpers.h"
#include "model/point.h"
#include "model/solver.h"
#include "measure/measure.cu"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cout << "Usage: <Algorithm> <Grid dimension> <Path to points file> <Path for hull export>\n";
        cout << "Algorithms:\n"
             << "\t-QH: Quickhull\n"
             << "\t-QHC: Quickhull with crawlers\n"
             << "\t-CO: ConcurrentHull\n"
             << "\t-M: Measure QH, QHC, and CO by running 10 times (results written to <Path for export>)\n"
             << endl;

        return 0;
    }

    string algorithm = string(argv[1]);
    unsigned gridDim = atoi(argv[2]);
    string pointsFile = string(argv[3]);
    string exportPath = string(argv[4]);

    Point *points = nullptr, *hull = nullptr;
    unsigned n, hullSize = 0;

    importPoints(&points, &n, pointsFile);

    cout << "Loaded " << n << " points." << endl;

    if (n < 1)
    {
        cout << "Exiting." << endl;
        return 0;
    }

    CConcurrentHull co(gridDim);
    CQuickHull qh;
    CQuickHullWithCrawlers qhc(gridDim);

    float elapsedTime;

    if (algorithm == "-QH")
        elapsedTime = runAlgorithm(qh, points, n, &hull, &hullSize);
    else if (algorithm == "-QHC")
        elapsedTime = runAlgorithm(qhc, points, n, &hull, &hullSize);
    else if (algorithm == "-CO")
        elapsedTime = runAlgorithm(co, points, n, &hull, &hullSize);
    else if (algorithm == "-M")
    {
        vector<CSolver *> solvers = {&qh, &qhc, &co};
        vector<vector<double>> times;
        measureTimes(solvers, points, n, 10, times);
        exportTimes(times, exportPath);
        return 0;
    }
    else
    {
        cout << "Invalid algorithm" << endl;
        return 1;
    }

    cout << "Total time: " << elapsedTime << " ms." << endl;

    cout << "Exporting " << hullSize << " points." << endl;
    exportPoints(hull, hullSize, exportPath);

    cudaFreeHost(points);
    cudaFreeHost(hull);

    checkForErrors();
};
