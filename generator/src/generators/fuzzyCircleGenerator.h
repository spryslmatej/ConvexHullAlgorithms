#pragma once

#include "../generator.h"

class CFuzzyCircleGenerator : public CGenerator
{
public:
    void generate(vector<Point> &points,
                  const unsigned count,
                  const unsigned maxVal) override;
};
