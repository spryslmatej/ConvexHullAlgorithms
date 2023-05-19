
#include <vector>
#include <chrono>
#include <fstream>

#include "../model/point.h"
#include "../model/solver.h"

#include "../quickhull/quickhull.h"
#include "../concurrent/concurrent.h"

using namespace std;

// returns measured time in ms
double runAlgorithm(CSolver &solver,
                    vector<Point> &points, vector<Point> &hull)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(),
                                          end;

    solver.solve(points, hull);

    end = std::chrono::steady_clock::now();

    auto elapsedTime = std::chrono::duration<double>(end - start);

    return elapsedTime.count() * 1000;
}

void measureTimes(const vector<CSolver *> solvers,
                  const vector<Point> &points,
                  const unsigned iterations,
                  vector<vector<double>> &times)
{
    vector<Point> tmpPoints, hull;
    for (CSolver *it : solvers)
    {
        vector<double> t;

        for (unsigned i = 0; i < iterations; i++)
        {
            tmpPoints = points;
            hull.clear();

            double elapsedTime = runAlgorithm(*it, tmpPoints, hull);
            t.push_back(elapsedTime);
        }

        times.push_back(t);
    }
}

void measureParallelization(const vector<CSolver *> solvers,
                            const vector<Point> &points)
{
    vector<Point> tmpPoints, hull;
    for (CSolver *it : solvers)
    {
        for (unsigned i = 0; i < 2; i++)
        {
            tmpPoints = points;
            hull.clear();

            omp_set_dynamic(0);
            if (i == 0)
                omp_set_num_threads(1);
            else
                omp_set_num_threads(omp_get_num_procs());

            double elapsedTime = runAlgorithm(*it, tmpPoints, hull);
            cout << i << " " << omp_get_max_threads() << " " << elapsedTime << endl;
        }
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
