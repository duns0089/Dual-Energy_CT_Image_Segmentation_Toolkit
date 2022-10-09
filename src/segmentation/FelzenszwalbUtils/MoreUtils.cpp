#pragma once
#include <string>
#include <iostream>
#include <functional>
#include <vector>
#include <cstdlib>

#include "DisjointForest.cpp"

#include "ImageSpace.h"
#include "Helpers.h"

std::vector<pixel_pointer> constructImagePixels(int3 num_image_voxels_xyz, float *original_image, float threshold, int &number_of_components, int method)
{
    unsigned int x, y, z, xyz;
    x = (unsigned int)num_image_voxels_xyz.x;
    y = (unsigned int)num_image_voxels_xyz.y;
    z = (unsigned int)num_image_voxels_xyz.z;
    xyz = x * y * z;

    std::vector<pixel_pointer> pixels(xyz);

    component_struct_pointer firstComponentStruct = std::make_shared<ComponentStruct>();
    auto previousComponentStruct = firstComponentStruct;
    int index;
    int componentCount = 0;

    double CurrentVolumeMSTMAxEdge = 0;

    component_pointer firstVolumeComponent;
    component_struct_pointer firstVolumeComponentStruct;

    for (int index_y = 0; index_y < y; index_y++)
    {
        for (int index_x = 0; index_x < x; index_x++)
        {
            for (int index_z = 0; index_z < z; index_z++)
            {
                index = INDEX3D(index_z, index_x, index_y, z, x);

                float value = original_image[index];
                if (method == 0)
                {
                    if (value >= threshold)
                    {
                        componentCount++;
                        number_of_components++;
                        component_pointer component = makeComponent(index_x, index_y, index_z, value);

                        component_struct_pointer newComponentStruct = std::make_shared<ComponentStruct>();
                        newComponentStruct->component = component;
                        newComponentStruct->previousComponentStruct = previousComponentStruct;
                        previousComponentStruct->nextComponentStruct = newComponentStruct;
                        component->parentComponentStruct = newComponentStruct;
                        previousComponentStruct = newComponentStruct;

                        pixels[index] = component->pixels.at(0);
                    }
                }

                else if (method == 1)
                {
                    if (value >= threshold)
                    {
                        componentCount++;
                        number_of_components++;
                        component_pointer component = makeComponent(index_x, index_y, index_z, value);

                        component_struct_pointer newComponentStruct = std::make_shared<ComponentStruct>();
                        newComponentStruct->component = component;
                        newComponentStruct->previousComponentStruct = previousComponentStruct;
                        previousComponentStruct->nextComponentStruct = newComponentStruct;
                        component->parentComponentStruct = newComponentStruct;
                        previousComponentStruct = newComponentStruct;

                        pixels[index] = component->pixels.at(0);
                    }
                    else if (firstVolumeComponent == nullptr) // first com n, make a new component as usual
                    {
                        number_of_components++;
                        componentCount++;

                        component_pointer component = makeComponent(index_x, index_y, index_z, value);

                        component->MSTMaxEdge = (double)value;

                        auto newComponentStruct = firstComponentStruct;

                        newComponentStruct->component = component;
                        newComponentStruct->previousComponentStruct = previousComponentStruct;
                        previousComponentStruct->nextComponentStruct = newComponentStruct;
                        component->parentComponentStruct = newComponentStruct;
                        previousComponentStruct = newComponentStruct;

                        firstVolumeComponent = component;
                        firstVolumeComponentStruct = newComponentStruct;

                        firstVolumeComponent->rank = 1;

                        pixels[index] = component->pixels.at(0);
                    }
                    else // no need to make a new component, just make the pixel and at it to the background component
                    {
                        std::shared_ptr<Pixel> pixel = std::make_shared<Pixel>();
                        pixel->x = index_x;
                        pixel->y = index_y;
                        pixel->z = index_z;
                        pixel->value = value;

                        pixel->parent = firstVolumeComponent->representative->parent; // not sure about this one

                        pixel->parentTree = firstVolumeComponent;

                        firstVolumeComponent->pixels.push_back(pixel);

                        if (value > CurrentVolumeMSTMAxEdge)
                        {
                            firstVolumeComponent->MSTMaxEdge = (double)threshold;
                        }

                        firstVolumeComponent->rank++;

                        pixels[index] = pixel;
                    }
                }
            }
        }
    }

    if (method == 0 || firstVolumeComponent == nullptr)
    {
        // get rid ofthe temporary firstComponentStruct from
        firstComponentStruct = firstComponentStruct->nextComponentStruct;
        firstComponentStruct->previousComponentStruct = nullptr;
    }
    else if (method == 1 && firstVolumeComponent != nullptr) // we have some volume at it to the other components
    {
        firstComponentStruct->previousComponentStruct = firstVolumeComponentStruct;
        firstComponentStruct->previousComponentStruct = nullptr;
    }

    return pixels;
}

unsigned int getEdgeArraySize(int3 num_image_voxels_xyz)
{
    unsigned int x, y, z;
    x = (unsigned int)num_image_voxels_xyz.x;
    y = (unsigned int)num_image_voxels_xyz.y;
    z = (unsigned int)num_image_voxels_xyz.z;

    unsigned int totalEdges = -(x * z) - (y * z) - (x * y) + (3 * x * y * z); // formula for edges in 6-con 3D lattice by x,y,z

    // 18-connected
    // totalEdges = -(5 * x * z) - (5 * y * z) - (5 * x * y) + (9 * x * y * z);

    return totalEdges;
}

std::vector<edge_pointer> setEdges(const std::vector<pixel_pointer> &pixels, int3 num_image_voxels_xyz, float threshold, int method)
{
    unsigned int x, y, z;
    x = (unsigned int)num_image_voxels_xyz.x;
    y = (unsigned int)num_image_voxels_xyz.y;
    z = (unsigned int)num_image_voxels_xyz.z;
    unsigned int edgeArraySize = getEdgeArraySize(num_image_voxels_xyz);

    std::vector<edge_pointer> edges(edgeArraySize);

    int edgeCount = 0;
    int edgesSkipped = 0;
    int allEdges = 0;

    const int connected_18[9][3] = {{1, 0, 0},
                                    {0, 1, 0},
                                    {0, 0, 1},
                                    {1, 1, 0},
                                    {1, 0, 1},
                                    {0, 1, 1},
                                    {0, -1, 1},
                                    {-1, 1, 0},
                                    {-1, 0, 1}};

    // boundary pixels can be included, pixels with on edges are checked and ignored later
    for (int index_y = 0; index_y < y; index_y++)
    {
        for (int index_x = 0; index_x < x; index_x++)
        {
            for (int index_z = 0; index_z < z; index_z++)
            {
                pixel_pointer presentPixel = pixels[INDEX3D(index_z, index_x, index_y, z, x)];

                if (presentPixel)
                {
                    if (method == 0 || (method == 1 && presentPixel->value >= threshold))
                    {
                        // iterature through each neighbour to connect to
                        for (int i = 0; i < 3; i++) // for 6 connected
                        // for (int i = 0; i < 9; i++) // 18-con
                        {
                            // coordinates of neighouring pixel to create an edge with
                            int x_component = index_x + connected_18[i][0];
                            int y_component = index_y + connected_18[i][1];
                            int z_component = index_z + connected_18[i][2];

                            if (x_component >= 0 && y_component >= 0 && z_component >= 0 && x_component < x && y_component < y && z_component < z)
                            {
                                int index_of_neighbouring_pixel = INDEX3D(z_component, x_component, y_component, z, x);

                                if (pixels[index_of_neighbouring_pixel])
                                {
                                    edges[edgeCount++] = createEdge(presentPixel, pixels[index_of_neighbouring_pixel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    edges.resize(edgeCount);

    return edges;
}

uint *addColorToSegmentation(component_struct_pointer componentStruct, int3 num_image_voxels_xyz, float *og_image)
{
    uint *segmentedImage = new uint[size(num_image_voxels_xyz)]();

    uint component_no = 0;

    // debug
    uint com_count = 0;
    uint pixel_count = 0;

    pixel_pointer previousPixel;

    do
    {
        com_count++;
        component_no++;

        int pixelCount = 0;

        int com_size = 0;

        for (const auto &pixel : componentStruct->component->pixels)
        {

            com_size++;
            pixelCount++;

            uint pixel_x = pixel->x;
            uint pixel_y = pixel->y;
            uint pixel_z = pixel->z;

            segmentedImage[INDEX3D(pixel_z, pixel_x, pixel_y, num_image_voxels_xyz.z, num_image_voxels_xyz.x)] = component_no;

            pixel_count++;
        }
        componentStruct = componentStruct->nextComponentStruct;

    } while (componentStruct);

    // debug
    // std::cout << "com count: " << com_count << std::endl;
    // std::cout << "total pixel count: " << pixel_count << std::endl;

    return segmentedImage;
}