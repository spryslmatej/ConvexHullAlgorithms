#pragma once

enum directions
{
    U = 0,
    RU,
    R,
    RD,
    D,
    LD,
    L,
    LU
};

void getStepsFromDirection(const directions dir, int &rowStep, int &colStep);
