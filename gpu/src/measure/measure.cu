
#include <vector>
#include <chrono>
#include <fstream>

#include "../model/point.h"
#include "../model/solver.h"

using namespace std;

// returns measured time in ms
float runAlgorithm(CSolver &solver,
                   const Point *points, const unsigned n,
                   Point **hull, unsigned *hullSize)
{
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    HANDLE_ERROR(cudaEventRecord(start, 0));

    solver.solve(points, n, hull, hullSize);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    return elapsedTime;
}

void measureTimes(const vector<CSolver *> solvers,
                  const Point *points,
                  const unsigned n,
                  const unsigned iterations,
                  vector<vector<double>> &times)
{
    Point *tmpPoints = (Point *)malloc(n * sizeof(Point)),
          *hull;
    unsigned hullSize;
    for (CSolver *it : solvers)
    {
        vector<double> t;

        for (unsigned i = 0; i < iterations; i++)
        {
            std::copy(points, points + n, tmpPoints);

            double elapsedTime = runAlgorithm(*it, tmpPoints, n, &hull, &hullSize);
            t.push_back(elapsedTime);

            cudaFreeHost(hull);
        }

        times.push_back(t);
    }
}

void exportTimes(const vector<vector<double>> &times, const string &path)
{
    ofstream qhStream = ofstream(path + "/Quickhull");
    ofstream qhcStream = ofstream(path + "/Quickhull with Crawlers");
    ofstream coStream = ofstream(path + "/Concurrent Hull");

    for (auto it : times.at(0))
    {
        qhStream << it << "\n";
    }
    qhStream << flush;

    for (auto it : times.at(1))
    {
        qhcStream << it << "\n";
    }
    qhcStream << flush;

    for (auto it : times.at(2))
    {
        coStream << it << "\n";
    }
    coStream << flush;

    qhStream.close();
    qhcStream.close();
    coStream.close();
}
