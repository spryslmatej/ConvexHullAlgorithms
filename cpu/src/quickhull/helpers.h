#pragma once

#include <vector>
#include <set>
#include <cmath>

#include "../model/point.h"
#include "../model/line.cpp"
#include "../model/triangle.cpp"

using namespace std;

void swapItems(vector<Point> &q, const unsigned x, const unsigned y);

//  MATH

double signPosition(const Point p1, const Point p2, const Point p3);

double getPointDistanceFromLine(const Line l, const Point z);

Point getPointFootToLine(const Line l,
                         const Point p);

//  tests if both points p and q are on the same side of line
bool arePointsOnSameSideOfLine(const Line l,
                               const Point p, const Point q);

bool isPointInTriangle(const Point pt, const Triangle t);