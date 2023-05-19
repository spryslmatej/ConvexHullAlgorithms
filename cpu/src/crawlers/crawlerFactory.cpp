#pragma once

#include <vector>

#include "crawler.h"
#include "../model/directions.cpp"

using namespace std;

struct CrawlerFactory
{
    static void createCrawlersFromPoint(const vector<vector<CrawlerSegment>> &segments,
                                        const unsigned i,
                                        const unsigned j,
                                        vector<Crawler> &crawlers)
    {
        vector<directions> dirs;
        getViableDirections(segments, i, j, dirs);

        for (auto it : dirs)
        {
            int rowStep = 0, colStep = 0;
            getStepsFromDirection(it, rowStep, colStep);
            crawlers.emplace_back(i, j, rowStep, colStep);
        }
    }

    static void getStepsFromDirection(const directions dir, int &rowStep, int &colStep)
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

    static void getViableDirections(const vector<vector<CrawlerSegment>> &segments,
                                    const unsigned i,
                                    const unsigned j,
                                    vector<directions> &dirs)
    {
        if (i == 0)
        {
            if (j == 0)
                dirs.push_back(RD);
            if (j == segments.at(0).size() - 1)
                dirs.push_back(LD);
            else
            {
                dirs.push_back(D);
                dirs.push_back(RD);
                dirs.push_back(LD);
            }
        }
        else if (i == segments.size() - 1)
        {
            if (j == 0)
                dirs.push_back(RU);
            if (j == segments.at(0).size() - 1)
                dirs.push_back(LU);
            else
            {
                dirs.push_back(L);
                dirs.push_back(RU);
                dirs.push_back(LU);
            }
        }
        else
        {
            if (j == 0)
            {
                dirs.push_back(R);
                dirs.push_back(RD);
                dirs.push_back(RU);
            }
            else if (j == segments.at(0).size() - 1)
            {
                dirs.push_back(L);
                dirs.push_back(LD);
                dirs.push_back(LU);
            }
        }
    }
};