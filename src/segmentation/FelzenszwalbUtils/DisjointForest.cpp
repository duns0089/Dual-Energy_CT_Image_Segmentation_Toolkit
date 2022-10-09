#pragma once

#include "DisjointForest.h"
#include <cmath>
#include <memory>

std::shared_ptr<Component> makeComponent(unsigned int x, unsigned int y, unsigned int z, float value)
{
    std::shared_ptr<Component> component = std::make_shared<Component>();
    std::shared_ptr<Pixel> pixel = std::make_shared<Pixel>();

    pixel->x = x;
    pixel->y = y;
    pixel->z = z;
    pixel->value = value;
    pixel->parent = static_cast<std::shared_ptr<Pixel>>(pixel);
    pixel->parentTree = static_cast<std::shared_ptr<Component>>(component);

    component->representative = static_cast<std::shared_ptr<Pixel>>(pixel);
    component->pixels.push_back(std::move(pixel));
    return component;
}

void setParentTree(const component_pointer &childTreePointer, const component_pointer &parentTreePointer)
{

    for (const auto &nodePointer : childTreePointer->pixels)
    {
        parentTreePointer->pixels.push_back(nodePointer);
        nodePointer->parentTree = parentTreePointer;
    }
}

double grayPixelDifference(const pixel_pointer &pixel1, const pixel_pointer &pixel2)
{
    return abs(pixel1->value - pixel2->value);
}

edge_pointer createEdge(const pixel_pointer &pixel1, const pixel_pointer &pixel2)
{
    edge_pointer edge = std::make_shared<Edge>();
    edge->weight = grayPixelDifference(pixel1, pixel2);
    edge->n1 = pixel1;
    edge->n2 = pixel2;
    return edge;
}

void mergeComponents(const component_pointer &x, const component_pointer &y, const double MSTMaxEdgeValue)
{
    if (x != y)
    {
        component_struct_pointer componentStruct;
        if (x->rank < y->rank)
        {
            x->representative->parent = y->representative;
            y->MSTMaxEdge = MSTMaxEdgeValue;
            setParentTree(x, y);
            componentStruct = x->parentComponentStruct;
        }
        else
        {
            if (x->rank == y->rank)
            {
                ++x->rank;
            }
            y->representative->parent = x->representative->parent;
            x->MSTMaxEdge = MSTMaxEdgeValue;

            setParentTree(y, x);
            componentStruct = y->parentComponentStruct;
        }
        if (componentStruct->previousComponentStruct)
        {

            componentStruct->previousComponentStruct->nextComponentStruct = componentStruct->nextComponentStruct;
        }
        if (componentStruct->nextComponentStruct)
        {

            componentStruct->nextComponentStruct->previousComponentStruct = componentStruct->previousComponentStruct;
        }
    }
}