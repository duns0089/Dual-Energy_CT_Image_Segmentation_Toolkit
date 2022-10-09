#pragma once
#include <queue>

#include "ImageSpace.h"
#include "Helpers.h"
#include "DIYTimer.cpp"

using sccl_segmented_image_type = uint32_t;

class SimpleCCL
{
private:
    int3 num_image_voxels_xyz;
    uint8_t *phantom_materials = nullptr;
    sccl_segmented_image_type *segmented_image = nullptr; // A mew image is created to save the segmented image
    std::queue<std::array<int, 3>> q; // Stores pixels that need to be checked, a pixel is added if it is neighbour and has same material id of current pixel being checked

    int pixel_connectivity = -1;
    int segment_count = -1;
    std::string output_path;

    unsigned int time_ms = 0;
    DIYTimer timer;

public:
    SimpleCCL(int3 num_image_voxels_xyz_, uint8_t *phantom_materials_, std::string output_path_)
    {
        num_image_voxels_xyz = num_image_voxels_xyz_;
        phantom_materials = phantom_materials_;
        output_path = output_path_;
    }

    void memsetSegmentedImage()
    {
        if (segmented_image != nullptr)
        {
            delete[] segmented_image;
        }
        segmented_image = new sccl_segmented_image_type[size(num_image_voxels_xyz)]();
    }

    void clear()
    {
        pixel_connectivity = -1;
        segment_count = -1;
        time_ms = 0;
        if (segmented_image != nullptr)
        {
            delete[] segmented_image;
        }
    }

    /**
     * @brief Segments a material phantom image onto an empty phantom. Uses a variant of the Connected Component Labelling approach.
     * Adapts a multivarient approach comparing material ID's instead of 0s and 1s.
     *
     * Sources:
     * Vincent, L., Vincent, L., &#38; Soille, P. (1991). Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations. 
     * IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(6), 583–598.
     * Similar algorithm shown in https://en.wikipedia.org/wiki/Connected-component_labeling#One_component_at_a_time
     *
     * @param[in] pixel_connectivity Specify pixel connectivity, '0' - 6 connected, '1' - 12 connected, '2' - 26 connected
     */
    void run(int pixel_connectivity_ = 0)
    {
        // Coordinates of neighbouring pixels, assuming relative pixel is 0,0,0
        int connected_6[6][3] = {{-1, 0, 0},
                                 {0, 0, -1},
                                 {0, 0, 1},
                                 {0, 1, 0},
                                 {1, 0, 0},
                                 {0, -1, 0}}; // Surfaces

        int connected_18[12][3] = {{0, -1, -1},
                                   {0, -1, 1},
                                   {0, 1, -1},
                                   {0, 1, 1},
                                   {1, 0, -1},
                                   {1, 0, 1},
                                   {1, -1, 0},
                                   {1, 1, 0},
                                   {-1, 0, -1},
                                   {-1, 0, 1},
                                   {-1, 1, 0},
                                   {-1, -1, 0}}; // Edges

        int connected_26[8][3] = {{1, -1, -1},
                                  {1, -1, 1},
                                  {1, 1, -1},
                                  {1, 1, 1},
                                  {-1, -1, -1},
                                  {-1, -1, 1},
                                  {-1, 1, -1},
                                  {-1, 1, 1}}; // Vertices

        sccl_segmented_image_type current_label = 1;
        pixel_connectivity = pixel_connectivity_;
        memsetSegmentedImage();

        timer = DIYTimer();

        for (int z = 0; z < num_image_voxels_xyz.z; z++) // boundary pixels can be included, pixels with on edges are checked and ignored later
        {
            for (int y = 0; y < num_image_voxels_xyz.y; y++)
            {
                for (int x = 0; x < num_image_voxels_xyz.x; x++)
                {
                    int current_mat_id = phantom_materials[INDEX3D(z, x, y, num_image_voxels_xyz.z, num_image_voxels_xyz.x)]; // pixels of same materail + are next to eachother are regioned
                    int label = segmented_image[INDEX3D(z, x, y, num_image_voxels_xyz.z, num_image_voxels_xyz.x)];            // each region is labelled, done by allocating every pixel in the region the label

                    if (current_mat_id != 0 && label == 0) // pixel and pixels neighbours need to be checked, mark as new region with label and added to queue
                    {
                        segmented_image[INDEX3D(z, x, y, num_image_voxels_xyz.z, num_image_voxels_xyz.x)] = current_label;
                        q.push({x, y, z});

                        while (!q.empty()) // Each new pixel in the same region is added to queue, queue finished = region complete
                        {
                            std::array<int, 3> front = q.front();
                            int front_x = front[0];
                            int front_y = front[1];
                            int front_z = front[2];

                            // 6 connected (faces)
                            for (int i1 = 0; i1 < 6; i1++)
                            {
                                int x_component = front_x + connected_6[i1][0];
                                int y_component = front_y + connected_6[i1][1];
                                int z_component = front_z + connected_6[i1][2];

                                if (x_component >= 0 && y_component >= 0 && z_component >= 0 && x_component < num_image_voxels_xyz.x && y_component < num_image_voxels_xyz.y && z_component < num_image_voxels_xyz.z) // if boundary pixels, avoids get seg fault
                                {
                                    checkPixel(x_component, y_component, z_component, current_mat_id, current_label);
                                }
                            }

                            // 18 connected minus 6 connected) (edges)
                            if (pixel_connectivity == 1 || pixel_connectivity == 2)
                            {
                                for (int i2 = 0; i2 < 12; i2++)
                                {
                                    int x_component = front_x + connected_18[i2][0];
                                    int y_component = front_y + connected_18[i2][1];
                                    int z_component = front_z + connected_18[i2][2];

                                    if (x_component >= 0 && y_component >= 0 && z_component >= 0 && x_component < num_image_voxels_xyz.x && y_component < num_image_voxels_xyz.y && z_component < num_image_voxels_xyz.z)
                                    {
                                        checkPixel(x_component, y_component, z_component, current_mat_id, current_label);
                                    }
                                }

                                // 26 connected minus 18 connected) (vertices)
                                if (pixel_connectivity == 2)
                                {
                                    for (int i3 = 0; i3 < 8; i3++)
                                    {
                                        int x_component = front_x + connected_26[i3][0];
                                        int y_component = front_y + connected_26[i3][1];
                                        int z_component = front_z + connected_26[i3][2];

                                        if (x_component >= 0 && y_component >= 0 && z_component >= 0 && x_component < num_image_voxels_xyz.x && y_component < num_image_voxels_xyz.y && z_component < num_image_voxels_xyz.z)
                                        {
                                            checkPixel(x_component, y_component, z_component, current_mat_id, current_label);
                                        }
                                    }
                                }
                            }
                            q.pop();
                        }
                        current_label++; // Region found, get label ready for next region found
                    }
                }
            }
        }
        time_ms = timer.finish();
        std::cout << "Time to segment: " << time_ms << " ms" << std::endl;

        segment_count = current_label - 1;
        std::cout << "Number of labels: " << segment_count << " (connectivity bool: " << pixel_connectivity << ")" << std::endl;
        return;
    }

    /**
     * @brief Method to check if the pixel at the supplied corrdinates belongs to a particular region of the same material but checking that it has a material ID that matches the supplied region material ID.
     * If also checks if the pixel is already labelled so it is not checked twice. If the pixel is not labelled and belongs to the region, the coordinates are added to a Queue.
     *
     * @param[in] x_component x value of pixel to check
     * @param[in] y_component y value of pixel to check
     * @param[in] z_component z value of pixel to check
     * @param[in] current_mat_id Material ID to check the pixel with
     * @param[in] current_label The label to give the checked pixel, if it matches the current_mat_id and has a current label of 0
     */
    void checkPixel(int x_component, int y_component, int z_component, int current_mat_id, sccl_segmented_image_type current_label)
    {
        int neighbour_mat_id = phantom_materials[INDEX3D(z_component, x_component, y_component, num_image_voxels_xyz.z, num_image_voxels_xyz.x)];
        int label = segmented_image[INDEX3D(z_component, x_component, y_component, num_image_voxels_xyz.z, num_image_voxels_xyz.x)];

        if (neighbour_mat_id == current_mat_id && label == 0) // pixel is apart of region and add it to queue so it's neighbours can be checked also
        {
            segmented_image[INDEX3D(z_component, x_component, y_component, num_image_voxels_xyz.z, num_image_voxels_xyz.x)] = current_label;
            q.push({x_component, y_component, z_component});
        }

        return;
    }

    void saveSegmentedImage()
    {
        std::string file_name;

        int x, y, z;
        x = num_image_voxels_xyz.x;
        y = num_image_voxels_xyz.y;
        z = num_image_voxels_xyz.z;

        switch (pixel_connectivity)
        {
        case 0:
            file_name = "SimpleCCL_6con_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit" + ".raw";
            break;
        case 1:
            file_name = "SimpleCCL_18con_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit" + ".raw";
            break;
        case 2:
            file_name = "SimpleCCL_26con_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + "_U32Bit" + ".raw";
            break;
        default:
            std::cout << "ERROR: no segmentation to save" << std::endl;
            break;
        }
        if (file_name.empty())
        {
            return;
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
            exit(1);
        }

        // Read data into buffer
        for (int height = 0; height < z; ++height)
        {
            for (int length = 0; length < y; ++length)
            {
                for (int width = 0; width < x; ++width)
                {
                    int index = INDEX3D(height, width, length, z, x);
                    file.write(reinterpret_cast<char *>(&segmented_image[index]), sizeof(sccl_segmented_image_type));
                }
            }
        }
        file.close();

        return;
    }

    int getSegmentCount()
    {
        if (segment_count == -1)
        {
            std::cout << "ERROR: segment count not set, segmentation has not run yet" << std::endl;
            return segment_count;
        }
        else
        {
            return segment_count;
        }
    }

    int getPixelConnectivity()
    {
        if (pixel_connectivity == -1)
        {
            std::cout << "ERROR: pixel connectivity not set, segmentation has not run yet" << std::endl;
            return pixel_connectivity;
        }
        else
        {
            return pixel_connectivity;
        }
    }

    sccl_segmented_image_type *getSegmentedImage()
    {
        return segmented_image;
    }
};
