#include "segment.h"

Segment::Segment() {}
Segment::Segment(unsigned b, unsigned e) : begin(b), end(e) {}

int Segment::size() { return end - begin; }

std::ostream &operator<<(std::ostream &out, const Segment &s)
{
    return out << "[" << s.begin << ", " << s.end << ")";
}
