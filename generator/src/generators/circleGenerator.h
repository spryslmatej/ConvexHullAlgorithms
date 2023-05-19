#pragma once

#include "../generator.h"

class CCircleGenerator : public CGenerator
{
public:
    void generate(vector<Point> &points,
                  const unsigned count,
                  const unsigned maxVal) override;
};
