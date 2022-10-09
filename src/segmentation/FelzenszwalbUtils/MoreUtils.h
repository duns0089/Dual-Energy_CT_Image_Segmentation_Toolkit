#pragma once
#include "ImageSpace.h"
#include "Helpers.h"

#include "DisjointForest.h"

std::vector<edge_pointer> setEdges(const std::vector<pixel_pointer> &pixels, int3 num_image_voxels_xyz, float threshold, int method);
uint *addColorToSegmentation(component_struct_pointer componentStruct, int3 num_image_voxels_xyz, float *og_image);
unsigned int getEdgeArraySize(int3 num_image_voxels_xyz);
std::vector<pixel_pointer> constructImagePixels(int3 num_image_voxels_xyz, float *original_image, float threshold, int &number_of_components, int method);
