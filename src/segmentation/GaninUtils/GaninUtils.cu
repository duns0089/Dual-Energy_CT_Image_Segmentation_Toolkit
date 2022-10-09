/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This application demonstrates an approach to the image segmentation
 * trees construction. It is based on Boruvka's MST algorithm.
 * Here's the complete list of references:
 * 1) V. Vineet et al, "Fast Minimum Spanning Tree for
 *    Large Graphs on the GPU";
 * 2) P. Felzenszwalb et al, "Efficient Graph-Based Image Segmentation";
 * 3) A. Ion et al, "Considerations Regarding the Minimum Spanning
 *    Tree Pyramid Segmentation Method".
 */

// System includes.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// STL includes.
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <list>
#include <deque>
#include <algorithm>

// Thrust library includes.
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/find.h>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

// Sample framework includes.
#include <helper_functions.h>
#include <helper_cuda.h>
#include <Helpers.h>

// Project includes.
// #include "common.cuh"

// Kernels.
#include "kernels.cuh"

using std::cin;
using std::cout;
using std::deque;
using std::endl;
using std::list;
using std::vector;

inline void WaitEnter()
{
    std::cout << "Press Enter to continue...";
    while (std::cin.get() != '\n')
    {
        // waiting
    }
}

// Very simple von Neumann middle-square prng.  rand() is different across
// various OS platforms, which makes testing and the output inconsistent.
int myrand(void)
{
    static int seed = 72191;
    char sq[22];

    seed *= seed;
    sprintf(sq, "%010d", seed);
    // pull the middle 5 digits out of sq
    sq[8] = 0;
    seed = atoi(&sq[3]);

    return seed;
}

// Simple memory pool class. It is nothing more than array of fixed-sized
// arrays.
template <typename T>
class DeviceMemoryPool
{
private:
    uint chunkSize_, chunkRawSize_;
    thrust::device_ptr<void> basePtr_;

    list<thrust::device_ptr<T>> chunks_;

public:
    // The parameters of the constructor are as follows:
    // 1) uint chunkSize --- size of the particular array;
    // 2) uint chunksCount --- number of fixed-sized arrays.
    DeviceMemoryPool(uint chunkSize, uint chunksCount) : chunkSize_(chunkSize)
    {
        // cout << "here" << endl;
        chunkRawSize_ = (chunkSize * sizeof(T) + 511) & ~511;

        try
        {
            basePtr_ = thrust::device_malloc(chunkRawSize_ * chunksCount);
        }
        catch (thrust::system_error &e)
        {
            cout << "Pool memory allocation failed (1) (" << e.what() << ")" << endl;
            exit(EXIT_FAILURE);
        }
        catch (thrust::system::detail::bad_alloc &e)
        {
            cout << "Pool memory allocation failed (2) (" << e.what() << ")" << endl;
            exit(EXIT_FAILURE);
        }

        for (uint chunkIndex = 0; chunkIndex < chunksCount; ++chunkIndex)
        {
            chunks_.push_back(thrust::device_ptr<T>(reinterpret_cast<T *>(static_cast<char *>(basePtr_.get()) + chunkRawSize_ * chunkIndex)));
        }
    }

    ~DeviceMemoryPool()
    {
        try
        {
            thrust::device_free(basePtr_);
        }
        catch (thrust::system_error &e)
        {
            cout << "Pool memory allocation failed (" << e.what() << ")"
                 << endl;
            exit(EXIT_FAILURE);
        }
    }

    // Returns an address of the first available array
    // in the memory pool.
    thrust::device_ptr<T> get()
    {
        thrust::device_ptr<T> ptr(chunks_.back());
        chunks_.pop_back();

        return ptr;
    }

    // Pushes an address stored in "ptr" to the list
    // of available arrays of the memory pool.
    // It should be noted that it is user who is responsible for returning
    // the previously requested memory to the appropriate pool.
    inline void put(const thrust::device_ptr<T> &ptr)
    {
        chunks_.push_back(ptr);
    }

    uint totalFreeChunks() const
    {
        return chunks_.size();
    }
};

// Graph structure.
struct Graph
{
    Graph() {}

    Graph(uint verticesCount, uint edgesCount) : vertices(verticesCount),
                                                 edges(edgesCount),
                                                 weights(edgesCount)
    {
    }

    // This vector stores offsets for each vertex in "edges" and "weights"
    // vectors. For example:
    // "vertices[0]" is an index of the first outgoing edge of vertex #0,
    // "vertices[1]" is an index of the first outgoing edge of vertex #1, etc.
    vector<uint> vertices;

    // This vector stores indices of endpoints of the corresponding edges.
    // For example, "edges[vertices[0]]" is the first neighbouring vertex
    // of vertex #0.
    vector<uint> edges;

    // This vector stores weights of the corresponding edges.
    vector<float> weights;
};

// Simple segmentation tree class.
// Each level of the tree corresponds to the segmentation.
// See "Level" class for the details.
class Pyramid
{
public:
    void addLevel(uint totalSuperNodes, uint totalNodes, thrust::device_ptr<uint> superVerticesOffsets, thrust::device_ptr<uint> verticesIDs)
    {
        levels_.push_back(Level(totalSuperNodes, totalNodes));
        levels_.back().buildFromDeviceData(superVerticesOffsets, verticesIDs);
    }

    uint levelsCount() const
    {
        return static_cast<uint>(levels_.size());
    }

    void dump(int3 num_image_voxels_xyz) const
    {

        uint total_x, total_y, total_z;
        total_x = num_image_voxels_xyz.x;
        total_y = num_image_voxels_xyz.y;
        total_z = num_image_voxels_xyz.z;

        uint levelIndex = 0;

        uint requiredDigitsCount = static_cast<uint>(log10(static_cast<float>(levelsCount()))) + 1;

        for (LevelsIterator level = levels_.rbegin(); level != levels_.rend(); ++level, ++levelIndex)
        {
            std::string filename = "GaninbMST_lvl" + std::to_string(levelIndex) + "_" + std::to_string(total_x) + "x" + std::to_string(total_y) + "x" + std::to_string(total_z) + "_U32Bit.raw";

            dumpLevel(level, num_image_voxels_xyz, filename);
        }
    }

private:
    // Level of the segmentation tree.
    class Level
    {
    public:
        Level(uint totalSuperNodes, uint totalNodes) : superNodesOffsets_(totalSuperNodes), nodes_(totalNodes)
        {
        }

        void buildFromDeviceData(
            thrust::device_ptr<uint> superVerticesOffsets,
            thrust::device_ptr<uint> verticesIDs)
        {
            checkCudaErrors(
                cudaMemcpy(&(superNodesOffsets_[0]),
                           superVerticesOffsets.get(),
                           sizeof(uint) * superNodesOffsets_.size(),
                           cudaMemcpyDeviceToHost));

            checkCudaErrors(
                cudaMemcpy(&(nodes_[0]),
                           verticesIDs.get(),
                           sizeof(uint) * nodes_.size(),
                           cudaMemcpyDeviceToHost));
        }

    private:
        friend class Pyramid;

        // The pair of the following vectors describes the
        // relation between the consecutive levels.
        // Consider an example. Let the index of the current level be n.
        // Then nodes of level #(n-1) with indices stored in
        // "nodes[superNodesOffsets_[0]]",
        // "nodes[superNodesOffsets_[0] + 1]",
        // ...,
        // "nodes[superNodesOffsets_[1] - 1]"
        // correspond to vertex #0 of level #n. An so on.
        vector<uint> superNodesOffsets_;
        vector<uint> nodes_;
    };

    typedef list<Level>::const_reverse_iterator LevelsIterator;

    // Dumps level to the file "level_n.ppm" where n
    // is index of the level. Segments are drawn in random colors.
    // void dumpLevel(LevelsIterator level, uint width, uint height, const char *filename) const
    void dumpLevel(LevelsIterator level, int3 num_image_voxels_xyz, std::string filename) const
    {

        uint total_x, total_y, total_z;
        total_x = num_image_voxels_xyz.x;
        total_y = num_image_voxels_xyz.y;
        total_z = num_image_voxels_xyz.z;

        deque<std::pair<uint, uint>> nodesQueue;

        uint totalSegments;

        {
            const vector<uint> &superNodesOffsets = level->superNodesOffsets_;
            const vector<uint> &nodes = level->nodes_;

            totalSegments = static_cast<uint>(superNodesOffsets.size());

            for (uint superNodeIndex = 0, nodeIndex = 0; superNodeIndex < superNodesOffsets.size(); ++superNodeIndex)
            {

                uint superNodeEnd = superNodeIndex + 1 < superNodesOffsets.size() ? superNodesOffsets[superNodeIndex + 1] : static_cast<uint>(nodes.size());

                for (; nodeIndex < superNodeEnd; ++nodeIndex)
                {
                    nodesQueue.push_back(std::make_pair(nodes[nodeIndex], superNodeIndex));
                }
            }
        }

        ++level;

        while (level != levels_.rend())
        {
            uint superNodesCount = static_cast<uint>(nodesQueue.size());

            const vector<uint> &superNodesOffsets = level->superNodesOffsets_;
            const vector<uint> &nodes = level->nodes_;

            while (superNodesCount--)
            {
                std::pair<uint, uint> currentNode = nodesQueue.front();
                nodesQueue.pop_front();

                uint superNodeBegin = superNodesOffsets[currentNode.first];

                uint superNodeEnd =
                    currentNode.first + 1 < superNodesOffsets.size() ? superNodesOffsets[currentNode.first + 1] : static_cast<uint>(nodes.size());

                for (uint nodeIndex = superNodeBegin; nodeIndex < superNodeEnd; ++nodeIndex)
                {
                    nodesQueue.push_back(std::make_pair(nodes[nodeIndex], currentNode.second));
                }
            }

            ++level;
        }

        vector<uint> colors(totalSegments);

        for (uint colorIndex = 0; colorIndex < totalSegments; ++colorIndex)
        {
            colors[colorIndex] = colorIndex + 1;
        }
        cout << "Total segments: " << totalSegments << endl;

        uint32_t *image = new uint32_t[total_x * total_y * total_z];

        while (!nodesQueue.empty())
        {

            std::pair<uint, uint> currentNode = nodesQueue.front();
            nodesQueue.pop_front();

            uint pixelIndex = currentNode.first;
            uint pixelSegment = currentNode.second;

            image[pixelIndex] = colors[pixelSegment];
        }

        saveSegmentedImage(image, num_image_voxels_xyz, filename);

        delete[] image;
    }

    void saveSegmentedImage(uint32_t *image, int3 num_image_voxels_xyz, std::string file_name) const
    {
        int x, y, z;
        x = num_image_voxels_xyz.x;
        y = num_image_voxels_xyz.y;
        z = num_image_voxels_xyz.z;

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
                    file.write(reinterpret_cast<char *>(&image[index]), sizeof(uint32_t));
                }
            }
        }
        file.close();
        cout << "saved " << file_name << endl;
        return;
    }

    std::string output_path = "output";

    list<Level> levels_;
};

// The class that encapsulates the main algorithm.
class SegmentationTreeBuilder
{
public:
    SegmentationTreeBuilder() : verticesCount_(0), edgesCount_(0) {}

    ~SegmentationTreeBuilder() {}

    // Repeatedly invokes the step of the algorithm
    // until the limiting segmentation is found.
    // Returns time (in ms) spent on building the tree.
    float run(const Graph &graph, Pyramid &segmentations, int3 num_image_voxels_xyz)
    {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        // Allocate required memory pools. We need just 4 types of arrays.
        MemoryPoolsCollection pools = {
            DeviceMemoryPool<uint>(static_cast<uint>(graph.vertices.size()), kUintVerticesPoolsRequired),
            DeviceMemoryPool<float>(static_cast<uint>(graph.vertices.size()), kFloatVerticesPoolsRequired),
            DeviceMemoryPool<uint>(static_cast<uint>(graph.edges.size()), kUintEdgesPoolsRequired),
            DeviceMemoryPool<float>(static_cast<uint>(graph.edges.size()), kFloatEdgesPoolsRequired)};

        // Initialize internal variables
        try
        {
            initalizeData(graph, pools);
        }
        catch (thrust::system_error &e)
        {
            cout << "Initialization failed (" << e.what() << ")" << endl;
            exit(EXIT_FAILURE);
        }

        // Run steps
        AlgorithmStatus status;

        try
        {
            do
            {
                status = invokeStep(pools, segmentations, num_image_voxels_xyz);

            } while (status != ALGORITHM_FINISHED);
        }
        catch (thrust::system_error &e)
        {
            cout << "Algorithm failed (" << e.what() << ")" << endl;
            exit(EXIT_FAILURE);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        return elapsedTime;
    }

private:
    void printMemoryUsage()
    {
        size_t availableMemory, totalMemory, usedMemory;

        cudaMemGetInfo(&availableMemory, &totalMemory);
        usedMemory = totalMemory - availableMemory;

        cout << "Device memory: used " << usedMemory
             << " available " << availableMemory
             << " total " << totalMemory << endl;
    }

    struct MemoryPoolsCollection
    {
        DeviceMemoryPool<uint> uintVertices;
        DeviceMemoryPool<float> floatVertices;
        DeviceMemoryPool<uint> uintEdges;
        DeviceMemoryPool<float> floatEdges;
    };

    static const uint kUintVerticesPoolsRequired = 8;
    static const uint kFloatVerticesPoolsRequired = 3;
    static const uint kUintEdgesPoolsRequired = 8;
    static const uint kFloatEdgesPoolsRequired = 4;

    void initalizeData(const Graph &graph, MemoryPoolsCollection &pools)
    {
        // Get memory for the internal variables
        verticesCount_ = static_cast<uint>(graph.vertices.size());
        edgesCount_ = static_cast<uint>(graph.edges.size());

        dVertices_ = pools.uintVertices.get();
        dEdges_ = pools.uintEdges.get();
        dWeights_ = pools.floatEdges.get();

        dOutputEdgesFlags_ = pools.uintEdges.get();

        // Copy graph to the device memory
        checkCudaErrors(cudaMemcpy(dVertices_.get(), &(graph.vertices[0]), sizeof(uint) * verticesCount_, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dEdges_.get(), &(graph.edges[0]), sizeof(uint) * edgesCount_, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dWeights_.get(), &(graph.weights[0]), sizeof(float) * edgesCount_, cudaMemcpyHostToDevice));

        thrust::fill(dOutputEdgesFlags_, dOutputEdgesFlags_ + edgesCount_, 0);
    }

    // static const uint kMaxThreadsPerBlock = 256;
    static const uint kMaxThreadsPerBlock = 1024;

    // Calculates grid parameters of the consecutive kernel calls
    // based on the number of elements in the array.
    void calculateThreadsDistribution(uint totalElements, uint &blocksCount, uint &threadsPerBlockCount)
    {
        if (totalElements > kMaxThreadsPerBlock)
        {
            blocksCount = (totalElements + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;

            threadsPerBlockCount = kMaxThreadsPerBlock;
        }
        else
        {
            blocksCount = 1;
            threadsPerBlockCount = totalElements;
        }
    }

    enum AlgorithmStatus
    {
        ALGORITHM_NOT_FINISHED,
        ALGORITHM_FINISHED
    };

    AlgorithmStatus invokeStep(MemoryPoolsCollection &pools, Pyramid &segmentations, int3 num_image_voxels_xyz)
    {
        // cout << "invokeStep()" << endl;
        uint blocksCount, threadsPerBlockCount;
        int x, y, z;
        x = num_image_voxels_xyz.x;
        y = num_image_voxels_xyz.y;
        z = num_image_voxels_xyz.z;

        calculateThreadsDistribution(edgesCount_, blocksCount, threadsPerBlockCount);
        dim3 gridDimsForEdges(blocksCount, 1, 1);
        dim3 blockDimsForEdges(threadsPerBlockCount, 1, 1);

        calculateThreadsDistribution(verticesCount_, blocksCount, threadsPerBlockCount);
        dim3 gridDimsForVertices(blocksCount, 1, 1);
        dim3 blockDimsForVertices(threadsPerBlockCount, 1, 1);

        thrust::device_ptr<uint> dEdgesFlags = pools.uintEdges.get();

        thrust::fill(dEdgesFlags, dEdgesFlags + edgesCount_, 0);

        // 1) Maring the MST edges: Finding minimum weighted edge

        // Mark the first edge for each vertex in "dEdgesFlags"
        markSegments<<<gridDimsForVertices, blockDimsForVertices, 0>>>(dVertices_.get(), dEdgesFlags.get(), verticesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("markSegments launch failed.");

        // Now find minimum edges for each vertex.
        thrust::device_ptr<uint> dMinScannedEdges = pools.uintEdges.get();
        thrust::device_ptr<float> dMinScannedWeights = pools.floatEdges.get();

        thrust::inclusive_scan_by_key(dEdgesFlags,
                                      dEdgesFlags + edgesCount_,
                                      thrust::make_zip_iterator(thrust::make_tuple(dWeights_, dEdges_)),
                                      thrust::make_zip_iterator(thrust::make_tuple(dMinScannedWeights, dMinScannedEdges)),
                                      thrust::greater_equal<uint>(),
                                      thrust::minimum<thrust::tuple<float, uint>>());

        // To make things clear.
        // Let "dEdgesFlags" denote groups of edges that
        // correspond to the same vertices. Then the last edge of each group
        // (in "dMinScannedEdges" and "dMinScannedWeights") is now minimal.

        // 2) Marking the MST edges: Finding and removing cycles //

        // Calculate a successor vertex for each vertex. A successor of the
        // vertex v is a neighbouring vertex connected to v
        // by the minimal edge.
        thrust::device_ptr<uint> dSuccessors = pools.uintVertices.get();

        getSuccessors<<<gridDimsForVertices, blockDimsForVertices, 0>>>(dVertices_.get(), dMinScannedEdges.get(), dSuccessors.get(), verticesCount_, edgesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("getSuccessors launch failed.");

        pools.uintEdges.put(dMinScannedEdges);
        pools.floatEdges.put(dMinScannedWeights);

        // Remove cyclic successor dependencies. Note that there can be only
        // two vertices in a cycle. See fig. 7 & fig. 8 in [1] for details.
        removeCycles<<<gridDimsForVertices, blockDimsForVertices, 0>>>(dSuccessors.get(), verticesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("removeCycles launch failed.");

        // 3) Graph Construction: Merging vertices //

        // Build up an array of startpoints for edges. As already stated,
        // each group of edges denoted by "dEdgesFlags"
        // has the same startpoint.
        thrust::device_ptr<uint> dStartpoints = pools.uintEdges.get();
        thrust::inclusive_scan(dEdgesFlags, dEdgesFlags + edgesCount_, dStartpoints);
        addScalar<<<gridDimsForEdges, blockDimsForEdges, 0>>>(dStartpoints.get(), -1, edgesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("addScalar launch failed.");

        // Shrink the chains of successors. New successors will eventually
        // represent superpixels of the new level.
        thrust::device_ptr<uint> dRepresentatives = pools.uintVertices.get();

        getRepresentatives<<<gridDimsForVertices, blockDimsForVertices, 0>>>(dSuccessors.get(), dRepresentatives.get(), verticesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("getRepresentatives launch failed.");

        swap(dSuccessors, dRepresentatives);

        pools.uintVertices.put(dRepresentatives);

        // 4) Graph Construction: Assigned ids to superpixels //

        // Group vertices by successors' indices.
        thrust::device_ptr<uint> dClusteredVerticesIDs = pools.uintVertices.get();

        thrust::sequence(dClusteredVerticesIDs, dClusteredVerticesIDs + verticesCount_);

        thrust::sort(
            thrust::make_zip_iterator(
                thrust::make_tuple(thrust::device_ptr<uint>(dSuccessors),
                                   thrust::device_ptr<uint>(dClusteredVerticesIDs))),
            thrust::make_zip_iterator(
                thrust::make_tuple(thrust::device_ptr<uint>(dSuccessors + verticesCount_),
                                   thrust::device_ptr<uint>(dClusteredVerticesIDs + verticesCount_))));

        // Mark those groups.
        thrust::device_ptr<uint> dVerticesFlags_ = pools.uintVertices.get();

        thrust::fill(dVerticesFlags_, dVerticesFlags_ + verticesCount_, 0);

        thrust::adjacent_difference(dSuccessors, dSuccessors + verticesCount_, dVerticesFlags_, thrust::not_equal_to<uint>());

        cudaMemset((void *)dVerticesFlags_.get(), 0, sizeof(uint));

        // Assign new indices to the successors (the indices of vertices
        // at the new level).
        thrust::device_ptr<uint> dNewVerticesIDs_ = pools.uintVertices.get();

        thrust::inclusive_scan(dVerticesFlags_, dVerticesFlags_ + verticesCount_, dNewVerticesIDs_);

        pools.uintVertices.put(dVerticesFlags_);

        // Now we can calculate number of resulting superpixels easily.
        uint newVerticesCount;
        cudaMemcpy(&newVerticesCount, (dNewVerticesIDs_ + verticesCount_ - 1).get(), sizeof(uint), cudaMemcpyDeviceToHost);
        ++newVerticesCount;

        // There are two special cases when we can stop our algorithm:
        // 1) number of vertices in the graph remained unchanged;
        // 2) only one vertex remains.
        if (newVerticesCount == verticesCount_)
        {
            cout << "No. of vertices remained unchanged, finishing" << endl;
            return ALGORITHM_FINISHED;
        }
        else if (newVerticesCount == 1)
        {
            thrust::device_ptr<uint> dDummyVerticesOffsets = pools.uintVertices.get();

            cudaMemset((void *)dDummyVerticesOffsets.get(), 0, sizeof(uint));

            thrust::device_ptr<uint> dDummyVerticesIDs = pools.uintVertices.get();

            thrust::sequence(dDummyVerticesIDs, dDummyVerticesIDs + verticesCount_);

            segmentations.addLevel(1, verticesCount_, dDummyVerticesOffsets, dDummyVerticesIDs);

            // cout << "Only one vertex remains, finishing" << endl;
            return ALGORITHM_FINISHED;
        }

        // 5) Graph Construction: Removing and forming the new edge list //

        // Calculate how old vertices IDs map to new vertices IDs.
        thrust::device_ptr<uint> dVerticesMapping = pools.uintVertices.get();

        getVerticesMapping<<<gridDimsForVertices, blockDimsForVertices, 0>>>(dClusteredVerticesIDs.get(), dNewVerticesIDs_.get(), dVerticesMapping.get(), verticesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("getVerticesMapping launch failed.");

        pools.uintVertices.put(dNewVerticesIDs_);
        pools.uintVertices.put(dClusteredVerticesIDs);
        pools.uintVertices.put(dSuccessors);

        // Invalidate self-loops in the reduced graph (the graph
        // produced by merging all old vertices that have
        // the same successor).
        invalidateLoops<<<gridDimsForEdges, blockDimsForEdges, 0>>>(dStartpoints.get(), dVerticesMapping.get(), dEdges_.get(), edgesCount_);
        // cudaDeviceSynchronize();
        getLastCudaError("invalidateLoops launch failed.");

        // Calculate various information about the surviving
        // (new startpoints IDs and IDs of edges) and
        // non-surviving/contracted edges (their weights).
        thrust::device_ptr<uint> dNewStartpoints = pools.uintEdges.get();
        thrust::device_ptr<uint> dSurvivedEdgesIDs = pools.uintEdges.get();

        calculateEdgesInfo<<<gridDimsForEdges, blockDimsForEdges, 0>>>(dStartpoints.get(), dVerticesMapping.get(), dEdges_.get(), dWeights_.get(), dNewStartpoints.get(), dSurvivedEdgesIDs.get(), edgesCount_, newVerticesCount);
        // cudaDeviceSynchronize();
        getLastCudaError("calculateEdgesInfo launch failed.");

        pools.uintEdges.put(dStartpoints);

        // Group that information by the new startpoints IDs.
        // Keep in mind that we want to build new (reduced) graph and apply
        // the step of the algorithm to that one. Hence we need to
        // preserve the structure of the original graph: neighbours and
        // weights should be grouped by vertex.
        thrust::sort(
            thrust::make_zip_iterator(
                thrust::make_tuple(dNewStartpoints,
                                   dSurvivedEdgesIDs)),
            thrust::make_zip_iterator(
                thrust::make_tuple(dNewStartpoints + edgesCount_,
                                   dSurvivedEdgesIDs + edgesCount_)));

        // Find the group of contracted edges.
        uint *invalidEdgesPtr =
            thrust::find_if(
                dNewStartpoints,
                dNewStartpoints + edgesCount_,
                IsGreaterEqualThan<uint>(newVerticesCount))
                .get();

        // Calculate how many edges there are in the reduced graph.
        uint validEdgesCount = static_cast<uint>(invalidEdgesPtr - dNewStartpoints.get());

        // Mark groups of edges corresponding to the same vertex in the
        // reduced graph.
        thrust::adjacent_difference(dNewStartpoints, dNewStartpoints + edgesCount_, dEdgesFlags, thrust::not_equal_to<uint>());

        cudaMemset((void *)dEdgesFlags.get(), 0, sizeof(uint));
        cudaMemset((void *)dEdgesFlags.get(), 1, 1);

        pools.uintEdges.put(dNewStartpoints);

        // 6) Graph Construction: Constructing the vertex list //

        // Now we are able to build the reduced graph. See "Graph"
        // class for the details on the graph's internal structure.

        // Calculate vertices' offsets for the reduced graph.
        thrust::copy_if(thrust::make_counting_iterator(0U), thrust::make_counting_iterator(validEdgesCount), dEdgesFlags, dVertices_, thrust::identity<uint>()).get();

        pools.uintEdges.put(dEdgesFlags);

        // Build up a neighbourhood for each vertex in the reduced graph
        // (this includes recalculating edges' weights).
        calculateThreadsDistribution(validEdgesCount, blocksCount, threadsPerBlockCount);
        dim3 newGridDimsForEdges(blocksCount, 1, 1);
        dim3 newBlockDimsForEdges(threadsPerBlockCount, 1, 1);

        thrust::device_ptr<uint> dNewEdges = pools.uintEdges.get();
        thrust::device_ptr<float> dNewWeights = pools.floatEdges.get();

        makeNewEdges<<<newGridDimsForEdges, newBlockDimsForEdges, 0>>>(dSurvivedEdgesIDs.get(), dVerticesMapping.get(), dEdges_.get(), dWeights_.get(), dNewEdges.get(), dNewWeights.get(), validEdgesCount);
        // cudaDeviceSynchronize();
        getLastCudaError("makeNewEdges launch failed.");

        swap(dEdges_, dNewEdges);
        swap(dWeights_, dNewWeights);

        pools.uintEdges.put(dNewEdges);
        pools.floatEdges.put(dNewWeights);

        pools.uintEdges.put(dSurvivedEdgesIDs);

        // The graph's reconstruction is now finished.

        // Build new level of the segmentation tree. It is a trivial task
        // as we already have "dVerticesMapping" that contains all
        // sufficient information about the vertices' transformations.
        thrust::device_ptr<uint> dVerticesIDs = pools.uintVertices.get();
        thrust::device_ptr<uint> dNewVerticesOffsets = pools.uintVertices.get();

        thrust::sequence(dVerticesIDs, dVerticesIDs + verticesCount_);

        thrust::sort_by_key(dVerticesMapping, dVerticesMapping + verticesCount_, dVerticesIDs);

        thrust::unique_by_key_copy(dVerticesMapping, dVerticesMapping + verticesCount_, thrust::make_counting_iterator(0), thrust::make_discard_iterator(), dNewVerticesOffsets);

        segmentations.addLevel(newVerticesCount, verticesCount_, dNewVerticesOffsets, dVerticesIDs);

        pools.uintVertices.put(dVerticesIDs);
        pools.uintVertices.put(dNewVerticesOffsets);
        pools.uintVertices.put(dVerticesMapping);

        // We can now safely set new counts for vertices and edges.
        verticesCount_ = newVerticesCount;
        edgesCount_ = validEdgesCount;

        return ALGORITHM_NOT_FINISHED;
    }

    uint verticesCount_;
    uint edgesCount_;

    thrust::device_ptr<uint> dVertices_;
    thrust::device_ptr<uint> dEdges_;
    thrust::device_ptr<float> dWeights_;

    thrust::device_ptr<uint> dOutputEdgesFlags_;
};

// inline float distance(const uchar3 &first, const uchar3 &second)
inline float distance(const float &first, const float &second)
{
    return abs(first - second);
}

inline int getSingleIndex(int3 num_image_voxels_xyz, int ix, int iy, int iz)
{
    int index = INDEX3D(iz, ix, iy, num_image_voxels_xyz.z, num_image_voxels_xyz.x);
    return index;
}

// Builds a net-graph for the image with 4-connected pixels.
// Builds a net-graph for the image with 4-connected pixels.
// void buildGraph(const vector<uchar3> &image, uint width, uint height, Graph &graph)
void buildGraph(float *image, int3 num_image_voxels_xyz, Graph &graph)
{
    uint connected_6[6][3] = {{0, 0, 1},
                              {0, 1, 0},
                              {1, 0, 0},
                              {-1, 0, 0},
                              {0, -1, 0},
                              {0, 0, -1}};

    uint connectedArrayLength = sizeof(connected_6) / sizeof(connected_6[0]);
    cout << "connectedArrayLength: " << connectedArrayLength << endl;

    uint total_x, total_y, total_z, totalNodes, totalEdges;
    total_x = num_image_voxels_xyz.x;
    total_y = num_image_voxels_xyz.y;
    total_z = num_image_voxels_xyz.z;
    totalNodes = size(num_image_voxels_xyz);
    totalEdges = (6 * total_x * total_y * total_z) - (2 * total_x * total_z) - (2 * total_y * total_z) - (2 * total_x * total_y); // formula for edges in 6-con 3D lattice by x,y,z

    graph.vertices.resize(totalNodes);
    graph.edges.reserve(totalEdges);
    graph.weights.reserve(totalEdges);

    uint edgesProcessed = 0;

    for (uint y = 0; y < total_y; ++y)
    {
        for (uint x = 0; x < total_x; ++x)
        {
            for (uint z = 0; z < total_z; ++z)
            {

                uint nodeIndex = getSingleIndex(num_image_voxels_xyz, x, y, z);

                const float &centerPixel = image[nodeIndex];

                graph.vertices[nodeIndex] = edgesProcessed;

                // 6 connected (faces)
                for (int i = 0; i < connectedArrayLength; i++) // double edged edges
                {
                    int x_component = x + connected_6[i][0];
                    int y_component = y + connected_6[i][1];
                    int z_component = z + connected_6[i][2];

                    if (x_component >= 0 && y_component >= 0 && z_component >= 0 && x_component < num_image_voxels_xyz.x && y_component < num_image_voxels_xyz.y && z_component < num_image_voxels_xyz.z) // if boundary pixels, avoids get seg fault
                    {
                        uint neighbourNode = getSingleIndex(num_image_voxels_xyz, x_component, y_component, z_component);

                        const float &neighbourPixel = image[neighbourNode];

                        graph.edges.push_back(neighbourNode);
                        graph.weights.push_back(distance(centerPixel, neighbourPixel));

                        ++edgesProcessed;
                    }
                }
                // }
            }
        }
    }

    graph.vertices.shrink_to_fit();
    graph.edges.shrink_to_fit();
    graph.weights.shrink_to_fit();
}