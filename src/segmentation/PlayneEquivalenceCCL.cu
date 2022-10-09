#pragma once
#include "ImageSpace.h"
#include "Helpers.h"
#include "DIYTimer.cpp"

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdint>
#include <chrono>

#include "PlayneEqDirectUtils/reduction.cuh"
#include "PlayneEqDirectUtils/cuda_utils.cuh"

using label_type = unsigned int;
using image_type = unsigned int;

// Device Functions
__global__ void init_labels(label_type *g_labels, image_type *g_image);
__global__ void resolve_labels(label_type *g_labels);
__global__ void label_reduction(label_type *g_labels, image_type *g_image);

// Image Size (Device Constant)
__constant__ unsigned int cX, cY, cZ;
__constant__ unsigned int pX, pY, cXYZ;

class PlayneEquivalenceCCL
{
private:
    uint8_t *phantom_materials = nullptr;
    label_type *segmented_image = nullptr;
    int3 num_image_voxels_xyz;

    unsigned int x, y, z;
    unsigned int px, py, xyz;

    label_type *d_labels = nullptr; // Note: The linearised index function of g_labels is not the same INDEX3D()

    image_type *h_image = nullptr;
    image_type *d_image = nullptr; 

    bool *d_changed = nullptr;

    std::string output_path;

    unsigned int time_ms = 0;
    DIYTimer timer; // Just time to segment the image
    DIYTimer initial_timer; // This timer will include the time it takes to allocate memory on device (GPU)

public:
    PlayneEquivalenceCCL(int3 num_image_voxels_xyz_, uint8_t *phantom_materials_, std::string output_path_)
    {
        phantom_materials = phantom_materials_;
        num_image_voxels_xyz = num_image_voxels_xyz_;

        x = (unsigned int)num_image_voxels_xyz_.x;
        y = (unsigned int)num_image_voxels_xyz_.y;
        z = (unsigned int)num_image_voxels_xyz_.z;
        px = x;
        py = x * y;
        xyz = x * y * z;

        output_path = output_path_;
    }

    /**
     * @brief Segments a material phantom image onto an empty phantom. Uses a variant of the Connected Component Labelling approach.
     * Adapts a multivarient approach comparing material ID's instead of 0s and 1s.
     *
     * This method uses NVIDIA CUDA to execute parallel computations on the GPU
     *
     * Sources:
     * Playne, D. P., &#38; Hawick, K. (2018). A new algorithm for parallel connected-component labelling on GPUs.
     * IEEE Transactions on Parallel and Distributed Systems, 296), 1217–1230.
     * https://ieeexplore.ieee.org/document/8274991
     * https://github.com/DanielPlayne/playne-equivalence-algorithm
     * https://github.com/FolkeV/CUDA_CCL
     */
    void run()
    {
        // Initialise device
        cudaSetDevice(0);

        initial_timer = DIYTimer();

        // Allocate host memory
        h_image = new image_type[xyz];
        for (int i = 0; i < xyz; ++i)
        {
            h_image[i] = phantom_materials[i];
        }

        cudaMalloc((void **)&d_labels, xyz * sizeof(label_type));
        cudaMalloc((void **)&d_image, xyz * sizeof(image_type));
        cudaMalloc((void **)&d_changed, sizeof(bool));

        // Copy host to device memory
        cudaMemcpyToSymbol(cX, &x, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(cY, &y, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(cZ, &z, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(cXYZ, &xyz, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(pX, &px, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(pY, &py, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
        cudaMemcpy(d_image, h_image, xyz * sizeof(image_type), cudaMemcpyHostToDevice);

        // Set block size - Uncomment desired block size, depends on hardware
        dim3 block(8, 8, 8);
        // dim3 block(32, 4, 4);
        // dim3 block(16, 16, 4);
        // dim3 block(4, 4, 4);
        // dim3 block(16, 8, 8);
        // dim3 block(10, 10, 10);
        // dim3 block(8, 8, 16);
        dim3 grid(ceil(x / (float)block.x), ceil(y / (float)block.y), ceil(z / (float)block.z));

        timer = DIYTimer();

        // Begin
        // Initialise labels
        init_labels<<<grid, block>>>(d_labels, d_image);
        cudaDeviceSynchronize();
        checkCUDAErrors();

        // Analysis
        resolve_labels<<<grid, block>>>(d_labels);
        cudaDeviceSynchronize();
        checkCUDAErrors();

        // Label Reduction
        label_reduction<<<grid, block>>>(d_labels, d_image);
        cudaDeviceSynchronize();
        checkCUDAErrors();

        // Analysis
        resolve_labels<<<grid, block>>>(d_labels);
        cudaDeviceSynchronize();
        checkCUDAErrors();

        time_ms = timer.finish();
        std::cout << "Time to segment (w/o device allocation): " << time_ms << " ms" << std::endl;
        std::cout << "Time to segment (w/ device allocation): " << initial_timer.finish() << " ms" << std::endl;

        // Set segmented image
        segmented_image = new label_type[xyz]();
        cudaMemcpy(segmented_image, d_labels, xyz * sizeof(label_type), cudaMemcpyDeviceToHost);
        saveSegmentedImage(segmented_image);

        // Print Number of Components
        printf("Number of Components: %u\n", count_components(segmented_image, xyz));

        // Delete memory
        delete[] h_image;
        delete[] segmented_image;

        cudaFree(d_labels);
        cudaFree(d_image);
        cudaFree(d_changed);

        return;
    }

    unsigned int count_components(const label_type *labels, const unsigned int K)
    {
        unsigned int count = 0;
        for (int k = 0; k < K; k++)
        {
            if (labels[k] == k)
            {
                count++;
            }
        }
        return count;
    }

    void saveSegmentedImage(label_type *image)
    {
        std::string file_name = "PlayneEquivalenceDirect_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit" + ".raw";

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
                    file.write(reinterpret_cast<char *>(&image[index]), sizeof(label_type));
                }
            }
        }
        file.close();

        return;
    }
};

//------------------------------------------------------------------------------------
// Device Functions
//--------------------------------------------------------------------------------------

// Initialise Kernel
__global__ void init_labels(label_type *g_labels, image_type *g_image)
{
    // Calculate index
    const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

    // Check Range
    if ((ix < cX) && (iy < cY) && (iz < cZ))
    {
        // Load image
        const image_type pzyx = __ldg(&g_image[INDEX3D(iz, ix, iy, cZ, cX)]);

        // Neighbour Connections
        const bool nzm1yx = (iz > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix, iy, cZ, cX)])) : false;
        const bool nzym1x = (iy > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix, iy - 1, cZ, cX)])) : false;
        const bool nzyxm1 = (ix > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix - 1, iy, cZ, cX)])) : false;

        const bool nzm1ym1x = ((iz > 0) && (iy > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix, iy - 1, cZ, cX)])) : false;
        const bool nzm1yxm1 = ((iz > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix - 1, iy, cZ, cX)])) : false;
        const bool nzym1xm1 = ((iy > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix - 1, iy - 1, cZ, cX)])) : false;

        const bool nzym1xp1 = ((iy > 0) && (ix < cX - 1)) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix + 1, iy - 1, cZ, cX)])) : false;
        const bool nzm1yxp1 = ((iz > 0) && (ix < cX - 1)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix + 1, iy, cZ, cX)])) : false;
        const bool nzm1yp1x = ((iz > 0) && (iy < cY - 1)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix, iy + 1, cZ, cX)])) : false;

        // Label
        label_type label;

        // Initialise Label
        label = (nzyxm1) ? (iz * pY + iy * pX + ix - 1) : (iz * pY + iy * pX + ix);
        label = (nzym1x) ? (iz * pY + (iy - 1) * pX + ix) : label;
        label = (nzm1yx) ? ((iz - 1) * pY + iy * pX + ix) : label;

        // Write to Global Memory
        g_labels[iz * pY + iy * pX + ix] = label; // Note: The linearised index function of g_labels is not the same INDEX3D()
    }
}

// Resolve Kernel
__global__ void resolve_labels(label_type *g_labels)
{
    // Calculate index
    const unsigned int id = ((blockIdx.z * blockDim.z) + threadIdx.z) * pY +
                            ((blockIdx.y * blockDim.y) + threadIdx.y) * pX +
                            ((blockIdx.x * blockDim.x) + threadIdx.x);

    // Check Range
    if (id < cXYZ)
    {
        // Resolve Label
        g_labels[id] = find_root(g_labels, g_labels[id]);
    }
}

// Label Reduction
__global__ void label_reduction(label_type *g_labels, image_type *g_image)
{
    // Calculate index
    const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int iz = (blockIdx.z * blockDim.z) + threadIdx.z;

    // Check Range
    if ((ix < cX) && (iy < cY) && (iz < cZ))
    {
        // Fetch Image
        const image_type pzyx = __ldg(&g_image[INDEX3D(iz, ix, iy, cZ, cX)]);

        // Compare Image Values
        const bool nzm1yx = (iz > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix, iy, cZ, cX)])) : false;
        const bool nzym1x = (iy > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix, iy - 1, cZ, cX)])) : false;
        const bool nzyxm1 = (ix > 0) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix - 1, iy, cZ, cX)])) : false;

        const bool nzym1xm1 = ((iy > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz, ix - 1, iy - 1, cZ, cX)])) : false;
        const bool nzm1yxm1 = ((iz > 0) && (ix > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix - 1, iy, cZ, cX)])) : false;
        const bool nzm1ym1x = ((iz > 0) && (iy > 0)) ? (pzyx == __ldg(&g_image[INDEX3D(iz - 1, ix, iy - 1, cZ, cX)])) : false;

        // Critical point conditions
        const bool cond_x = nzyxm1 && ((nzm1yx && !nzm1yxm1) || (nzym1x && !nzym1xm1));
        const bool cond_y = nzym1x && nzm1yx && !nzm1ym1x;

        // Get label
        unsigned int label1 = (cond_x || cond_y) ? g_labels[iz * pY + iy * pX + ix] : 0;

        // Y - Neighbour
        if (cond_y)
        {
            // Get neighbouring label
            label_type label2 = g_labels[iz * pY + (iy - 1) * pX + ix];

            // Reduction
            label1 = reduction(g_labels, label1, label2);
        }

        // X - Neighbour
        if (cond_x)
        {
            // Get neighbouring label
            label_type label2 = g_labels[iz * pY + iy * pX + ix - 1];

            // Reduction
            label1 = reduction(g_labels, label1, label2);
        }
    }
}