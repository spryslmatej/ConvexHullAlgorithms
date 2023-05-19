#include <iostream>
#include <vector>
#include <set>
#include <fstream>

#include "omp.h"

#include "model/point.h"
#include "model/solver.h"

#include "quickhull/quickhull.h"
#include "quickhull/quickhullWithCrawlers.h"
#include "chan/chan.h"
#include "graham/graham.h"
#include "jarvis/jarvis.h"
#include "concurrent/concurrent.h"

#include "utilities/export.cpp"
#include "utilities/import.cpp"

#include "measure/measure.cpp"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cout << "Usage: <Algorithm> <Grid dimension> <Path to points file> <Path for export> (<Thread num>)" << endl;
        cout << "Algorithms:\n"
             << "\t-G: Graham\n"
             << "\t-J: Jarvis\n"
             << "\t-QH: Quickhull\n"
             << "\t-QHC: Quickhull with crawlers\n"
             << "\t-CH: Chan\n"
             << "\t-CO: ConcurrentHull\n"
             << "\t-M: Measure QH, QHC, and CO by running 10 times (results written to <Path for export>)\n"
             << "\t-P: Measure QH and CO parallelization (results printed)\n"
             << endl;
        return 0;
    }

    omp_set_dynamic(0);
    if (argc == 6)
        omp_set_num_threads(atoi(argv[5]));
    else
        omp_set_num_threads(omp_get_num_procs());

    string algorithm = string(argv[1]);
    unsigned gridDim = atoi(argv[2]);
    string pointsFile = string(argv[3]);
    string exportPath = string(argv[4]);

    vector<Point> points = {};
    vector<Point> hull = {};
    double elapsedTime;

    importVector(points, pointsFile);

    cout << "Loaded " << points.size() << " points." << endl;

    CGrahamScan gr;
    CJarvisMarch jm;
    CQuickHull qh;
    CQuickHullWithCrawlers qhc(gridDim);
    CChan ch(gridDim);
    CConcurrentHull co(gridDim);

    if (algorithm == "-G")
        elapsedTime = runAlgorithm(gr, points, hull);
    else if (algorithm == "-J")
        elapsedTime = runAlgorithm(jm, points, hull);
    else if (algorithm == "-QH")
        elapsedTime = runAlgorithm(qh, points, hull);
    else if (algorithm == "-QHC")
        elapsedTime = runAlgorithm(qhc, points, hull);
    else if (algorithm == "-CH")
        elapsedTime = runAlgorithm(ch, points, hull);
    else if (algorithm == "-CO")
        elapsedTime = runAlgorithm(co, points, hull);
    else if (algorithm == "-M")
    {
        vector<CSolver *> solvers = {&qh, &qhc, &co};
        vector<vector<double>> times;
        measureTimes(solvers, points, 10, times);
        exportTimes(times, exportPath);
        return 0;
    }
    else if (algorithm == "-P")
    {
        vector<CSolver *> solvers = {&qh, &qhc, &co};
        measureParallelization(solvers, points);
        return 0;
    }
    else
    {
        cout << "Invalid algorithm" << endl;
        return 1;
    }

    cout << "Total elapsed time: " << elapsedTime << " ms." << endl;
    cout << "Exporting " << hull.size() << " points." << endl;
    cout << endl;

    exportVector(hull, exportPath);
}
