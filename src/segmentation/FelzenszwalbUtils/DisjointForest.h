#pragma once
#include <vector>
#include <functional>
#include <memory>

struct Component;
struct ComponentStruct;
struct Pixel;
struct Edge;

using component_pointer = std::shared_ptr<Component>;
using component_struct_pointer = std::shared_ptr<ComponentStruct>;
using pixel_pointer = std::shared_ptr<Pixel>;
using edge_pointer = std::shared_ptr<Edge>;

struct Pixel
{
    component_pointer parentTree;
    pixel_pointer parent;
    float value;
    unsigned int x, y, z;
};

struct Component
{
    component_struct_pointer parentComponentStruct;
    std::vector<pixel_pointer> pixels;
    int rank = 0;
    pixel_pointer representative;
    double MSTMaxEdge = 0;
};

struct Edge
{
    // double weight;
    float weight;
    pixel_pointer n1;
    pixel_pointer n2;
};

struct ComponentStruct
{
    component_struct_pointer previousComponentStruct = nullptr;
    component_pointer component{};
    component_struct_pointer nextComponentStruct = nullptr;
};

double grayPixelDifference(const pixel_pointer &pixel1, const pixel_pointer &pixel2);
edge_pointer createEdge(const pixel_pointer &pixel1, const pixel_pointer &pixel2); ///, const std::function<double(pixel_pointer, pixel_pointer)> &edgeDifferenceFunction);
component_pointer makeComponent(unsigned int x, unsigned int y, unsigned int z, float value);

void mergeComponents(const component_pointer &x, const component_pointer &y, double MSTMaxEdgeValue);

// /// debug
// bool mergeComponents(const component_pointer &x, const component_pointer &y, const double MSTMaxEdgeValue);
// bool check_pixel_relationship(pixel_pointer p1, pixel_pointer p2);
// //
