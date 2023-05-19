#include "directions.h"

void getStepsFromDirection(const directions dir, int &rowStep, int &colStep)
{
    switch (dir)
    {
    case U:
        rowStep = -1;
        break;
    case RU:
        rowStep = -1;
        colStep = 1;
        break;
    case R:
        colStep = 1;
        break;
    case RD:
        rowStep = 1;
        colStep = 1;
        break;
    case D:
        rowStep = 1;
        break;
    case LD:
        rowStep = 1;
        colStep = -1;
        break;
    case L:
        colStep = -1;
        break;
    case LU:
        rowStep = -1;
        colStep = -1;
        break;
    }
}
