#pragma once
#include "ImageSpace.h"
#include "Helpers.h"
#include "DIYTimer.cpp"

#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <queue>

// #include "DisjointForest.h"
// #include "segmentation.h"
// #include "MoreUtils.h"
#include "DisjointForest.cpp"
#include "segmentation.cpp"
#include "MoreUtils.cpp"

using segmented_image_type_FMST = uint;

class FelzenszwalbMST
{
private:
    segmented_image_type_FMST *segmented_image = nullptr;
    float *reconstruction = nullptr; // original image
    int3 num_image_voxels_xyz;
    unsigned int x, y, z, xyz;

    std::vector<pixel_pointer> pixels;
    std::vector<edge_pointer> edges;
    std::vector<edge_pointer> *d_edges_vector;

    std::string output_path;

    unsigned int time_ms = 0;
    DIYTimer timer;

    unsigned int sort_time_ms = 0;
    DIYTimer sort_timer;

    float gaussianBlur, kValue;
    int minimumComponentSize;

    int method;
    float threshold;

public:
    FelzenszwalbMST(int3 num_image_voxels_xyz_, float *original_image, std::string output_path_)
    {
        num_image_voxels_xyz = num_image_voxels_xyz_;
        reconstruction = original_image;

        output_path = output_path_;

        x = (unsigned int)num_image_voxels_xyz_.x;
        y = (unsigned int)num_image_voxels_xyz_.y;
        z = (unsigned int)num_image_voxels_xyz_.z;
        xyz = x * y * z;
    }

    void memsetMate()
    {
        if (segmented_image != nullptr)
        {
            delete[] segmented_image;
        }
        segmented_image = new segmented_image_type_FMST[size(num_image_voxels_xyz)]();

        if (!pixels.empty())
        {
            pixels.clear();
        }
        if (!edges.empty())
        {
            edges.clear();
        }
    }

    /**
     * @brief Minimal Spanning Tree (Kruskal) Segmentation for greyscale CT data.
     *
     * Three use cases:
     * Method = 0, Threshold = 0 [Regular MST, doesn't ignore volume voxels]
     * Method = 0, Threshold != 0 [Ignores volume voxels completely]
     * Method = 1, Threshold != 0 [Ignores volume voxels when creating graph but they are given their own component that other non-volume pixels could join]
     *
     * Sources:
     * https://iammohitm.github.io/Graph-Based-Image-Segmentation/
     * Felzenszwalb, P. F., &#38; Huttenlocher, D. P. (2004). Efficient graph-based image segmentation. 
     * International Journal of Computer Vision, 59(2), 167–181.

     * @param[in] method_ 0 or 1. 0 Inlcudes the volume unless a threshold is include. If a threshold is include it completely ignores volume.
     * @param[in] threshld_ If method = 0, volume below threshold is complete ignore, else, volume is given it's own component
     */
    void run(int method_ = 0, float threshold_ = 0)
    {
        method = method_;
        threshold = threshold_;

        memsetMate();

        timer = DIYTimer();

        std::cout << "1. Constructing pixels... method: " << method << ", threshold: " << threshold << std::endl;

        int component_count = 0;
        pixels = constructImagePixels(num_image_voxels_xyz, reconstruction, threshold, component_count, method);

        std::cout << "Pixels size: " << pixels.size() << std::endl;
        std::cout << "Pixels capacity: " << pixels.capacity() << std::endl;
        std::cout << "Number of components: " << component_count << std::endl;

        std::cout << "\n2. Contructing edges... " << std::endl;

        edges = setEdges(pixels, num_image_voxels_xyz, threshold, method);

        std::cout << "Number of edges: " << std::to_string(edges.size()) << std::endl;

        std::cout << "\n3. Sorting edges... (std::sort) " << std::endl;

        std::sort(edges.begin(), edges.end(), [](const edge_pointer &e1, const edge_pointer &e2)
                  { return e1->weight < e2->weight; });

        std::cout << "\n4. Segmenting now... " << std::endl;

        minimumComponentSize = 20;
        kValue = 2;

        segmentImage(edges, component_count, minimumComponentSize, kValue);

        std::cout << "\n5. Getting segmented image... " << std::endl;

        pixel_pointer first_non_null_pixel;
        uint index = 0;
        while (!first_non_null_pixel)
        {
            first_non_null_pixel = pixels[index++];
        }

        auto firstComponentStruct = first_non_null_pixel->parentTree->parentComponentStruct;

        while (firstComponentStruct->previousComponentStruct)
        {
            firstComponentStruct = firstComponentStruct->previousComponentStruct;
        }

        // TESTING TO SEE IF NOT NEEDED
        // // int com = 0;
        // component_struct_pointer next = firstComponentStruct->nextComponentStruct;
        // while (next)
        // {
        //     // com++;
        //     next = next->nextComponentStruct;
        // }
        // // std::cout << com << std::endl;
        // // return;
        // // std::cout << "com first struct addy 2.0 " << firstComponentStruct << std::endl;

        segmented_image = addColorToSegmentation(firstComponentStruct, num_image_voxels_xyz, reconstruction);

        std::cout << "\n6. Saving segmented image... " << std::endl;

        saveSegmentedImage();

        time_ms = timer.finish(); // weird error

        std::cout << "Method: " << method << ", Threshold: " << threshold << std::endl;
        std::cout << "Time to segment: " << time_ms << " ms \n"
                  << std::endl;
    }

    void saveSegmentedImage()
    {
        std::string file_name;
        if (threshold != 0)
        {
            file_name = "FelzenszwalbMST_6con_mc" + std::to_string(minimumComponentSize) + "_k" + std::to_string((int)kValue) + "_t_m" + std::to_string(method) + "_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit.raw";
        }
        else
        {
            file_name = "FelzenszwalbMST_6con_mc" + std::to_string(minimumComponentSize) + "_k" + std::to_string((int)kValue) + "_m" + std::to_string(method) + "_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit.raw";
        }

        // Define the specific file to open based on the view
        std::ostringstream filename;
        filename << output_path << "/" << file_name;

        std::string file_open;
        file_open = filename.str();

        // Open File and ensure it can be opened
        std::ofstream file(file_open, std::ios::binary);
        if (!file.is_open())
        {
            std::cout << "ERROR: Can't create or access " << file_open << "." << std::endl;
        }

        // Read data into buffer
        for (int length = 0; length < y; ++length)
        {
            for (int width = 0; width < x; ++width)
            {
                for (int height = 0; height < z; ++height)
                {
                    int index = INDEX3D(height, width, length, z, x);
                    file.write(reinterpret_cast<char *>(&segmented_image[index]), sizeof(segmented_image_type_FMST));
                }
            }
        }

        return;
    }
};