#include "height_filter.h"

std::vector<Point> height_filter(const std::vector<Point>& pts, double min_z)
{
    std::vector<Point> filtered;
    filtered.reserve(pts.size());

    for (const auto& p : pts) {
        if (p.z > min_z) {
            filtered.emplace_back(p);
        }
    }

    return filtered;
}
