#pragma once

#include <optional>
#include <stack>

#include "../model/point.h"

using namespace std;

class DualStack
{
private:
    stack<Point> s;
    optional<Point> pTop = {}, pNextToTop = {};
    unsigned count = 0;

public:
    Point top() const
    {
        if (pTop)
            return pTop.value();
        else
            throw "No item on top.";
    }

    Point nextToTop() const
    {
        if (pNextToTop)
            return pNextToTop.value();
        else
            throw "No item next to top.";
    }

    void pop()
    {
        if (count == 0)
            return;
        else if (count == 1)
        {
            pTop = {};
        }
        else if (count == 2)
        {
            pTop = pNextToTop;
            pNextToTop = {};
        }
        else
        {
            pTop = pNextToTop;
            pNextToTop = s.top();
            s.pop();
        }

        count--;
    }

    void push(const Point p)
    {
        if (count == 0)
            pTop = p;
        else if (count == 1)
        {
            pNextToTop = pTop;
            pTop = p;
        }
        else
        {
            s.push(pNextToTop.value());
            pNextToTop = pTop;
            pTop = p;
        }

        count++;
    }

    unsigned size() const
    {
        return count;
    }

    bool empty() const
    {
        return count == 0;
    }
};
