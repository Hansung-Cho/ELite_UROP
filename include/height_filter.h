#pragma once
#include <vector>

struct Point {
    double x, y, z;

    Point() = default;
    Point(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

std::vector<Point> height_filter(const std::vector<Point>& pts, double min_z);