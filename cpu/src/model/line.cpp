#pragma once

#include "point.h"

struct Line
{
    Point x, y;

    Line(const Point q, const Point w)
    {
        if (q.X < w.X ||
            (q.X == w.X && q.Y < w.Y))
        {
            x = q;
            y = w;
        }
        else
        {
            x = w;
            y = q;
        }
    }
};