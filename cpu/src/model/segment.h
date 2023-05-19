#pragma once

#include <ostream>

using namespace std;

struct Segment
{
    unsigned begin, end;

    Segment();
    Segment(unsigned b, unsigned e);

    int size();
};

std::ostream &operator<<(std::ostream &out, const Segment &s);