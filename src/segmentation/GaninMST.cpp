// CT Images Segmentation Algorithm Based on Graph Cuts - Privalov et al.
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

#include "GaninUtils/GaninUtils.cu"

using segmented_image_type_GMST = unsigned int;

class GaninMST
{

public:
    GaninMST(int3 num_image_voxels_xyz_, float *original_image, std::string output_path_)
    {
        num_image_voxels_xyz = num_image_voxels_xyz_;
        reconstruction = original_image;

        output_path = output_path_;
    }

    void run()
    {
        timer = DIYTimer();

        cudaSetDevice(0);

        std::cout << "\n1. Building Graph... \n"
                  << std::flush;

        Graph graph;
        buildGraph(reconstruction, num_image_voxels_xyz, graph);

        cout << "\n2. Building segmentation tree... \n"
             << std::flush;

        Pyramid segmentations;

        SegmentationTreeBuilder algo;

        seg_timer = DIYTimer();

        elapsedTime = algo.run(graph, segmentations, num_image_voxels_xyz);

        std::cout << "...done in " << elapsedTime << " ms\n"
                  << std::flush;

        seg_time_ms = seg_timer.finish();
        std::cout << "time no dump: " << seg_time_ms << " ms (cpu clock, just seg) =============================" << std::endl;

        cout << "\n3. Dumping levels for each tree..." << endl;

        segmentations.dump(num_image_voxels_xyz);

        time_ms = timer.finish();
        std::cout << "time w/ dump: " << time_ms << " ms ======================================================" << std::endl;
    }

private:
    segmented_image_type_GMST *segmented_image = nullptr;
    float *reconstruction = nullptr; // original image
    int3 num_image_voxels_xyz;

    std::string output_path;

    float elapsedTime; // return from segmentation run
    unsigned int time_ms = 0;
    DIYTimer timer;

    unsigned int seg_time_ms = 0;
    DIYTimer seg_timer;

    void fillSegmentedImage()
    {
        segmented_image = new segmented_image_type_GMST[size(num_image_voxels_xyz)];

        for (int i = 0; i < size(num_image_voxels_xyz); ++i)
        {
            segmented_image[i] = 1;
        }
    }
};
