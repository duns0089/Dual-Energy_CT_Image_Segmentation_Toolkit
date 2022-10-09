#include <fstream>
#include <iostream>

#include "ImageSpace.h"
#include "nlohmann/json.hpp"

// Segmentation Classes
#include "SimpleCCL.cpp"
#include "PlayneEquivalenceCCL.cu"
#include "FelzenszwalbMST.cpp"
#include "GaninMST.cpp"

using namespace std;

nlohmann::json read_json_file(string path)
{
    cout << path << endl;
    ifstream file(path);

    return nlohmann::json::parse(file);
}

void read_reconstruction(int3 num_image_voxels_xyz, float *phantom_reconstruction, string file_name)
{
    // Use FILE functions to pass data
    ifstream phantom_file(file_name, ios::binary);
    if (!phantom_file.is_open())
    {
        cout << "Could not open reconstruction" << endl;
        exit(1);
    }

    // // Need to read phantom and check if size matches the info file // ----- > commented out as some tests only used half of the image
    // // Verify that the size from the json matches the phantom file size
    // if (fs::file_size(file_name.c_str()) != sizeof(float) * size(num_image_voxels_xyz))
    // // if (fs::file_size(file_name.c_str()) != sizeof(uint8_t) * size(num_image_voxels_xyz))
    // {
    //     cout << "Read failed 1" << endl;
    //     exit(1);
    // }

    // Read data into buffer
    if (phantom_reconstruction == nullptr)
    {
        cout << "Read failed 1" << endl;
        exit(1);
    }
    phantom_file.read(reinterpret_cast<char *>(phantom_reconstruction), sizeof(float) * size(num_image_voxels_xyz));
    phantom_file.close();
}

void save_reconstruction(int3 num_image_voxels_xyz, float *image, string output_path)
{

    int x, y, z;
    x = num_image_voxels_xyz.x;
    y = num_image_voxels_xyz.y;
    z = num_image_voxels_xyz.z;

    string file_name = "OriginalRecon_" + to_string(x) + "x" + to_string(y) + "x" + to_string(z) + "_32BitReal" + ".raw";

    // Define the specific file to open based on the view
    ostringstream filename;
    filename << output_path << "/" << file_name;

    string file_open;
    file_open = filename.str();

    // Open File and ensure it can be opened
    ofstream file(file_open, ios::binary);
    if (!file.is_open())
    {
        cout << "ERROR: Can't create or access " << file_open << "." << endl;
    }

    // Read data into buffer
    for (int length = 0; length < y; ++length)
    {
        for (int width = 0; width < x; ++width)
        {
            for (int height = 0; height < z; ++height)
            {
                int index = INDEX3D(height, width, length, num_image_voxels_xyz.z, num_image_voxels_xyz.x);

                file.write(reinterpret_cast<char *>(&image[index]), sizeof(float));
            }
        }
    }

    file.close();

    return;
}

int main(int argc, char **argv)
{
    int3 num_image_voxels_xyz;
    string output_path = "output";

    // variables if suuped config and material json
    nlohmann::json config, materials;
    ImageSpace image;
    uint8_t *phantom_materials;

    // variables if supplied .raw and dimensions
    unsigned int x, y, z;
    string reconstruction_path;
    float *phantom_reconstruction;

    cout << "\n----------------------------------------------------" << endl;

    if (argv[1] != "-r" && argc < 3)
    {
        cout << "Config and material JSON paths required, or" << endl;
        cout << "Give path to reconstruction (.raw) and x y z" << endl;
        return -1;
    }
    else if (strcmp(argv[1], "-r") == 0) // && argc < 6)
    {
        cout << "Using RAW recontruction and supplied X,Y,Z..." << endl;
        reconstruction_path = argv[2];
        x = stoi(argv[3]);
        y = stoi(argv[4]);
        z = stoi(argv[5]);

        num_image_voxels_xyz = make_int3(x, y, z);
        phantom_reconstruction = new float[size(num_image_voxels_xyz)]();

        cout << "num_image_voxels_xyz = [" << num_image_voxels_xyz.x << ", " << num_image_voxels_xyz.y << ", " << num_image_voxels_xyz.z << "]" << endl;
        cout << "Total voxels: " << size(num_image_voxels_xyz) << endl;

        // Read in reconstruction
        read_reconstruction(num_image_voxels_xyz, phantom_reconstruction, reconstruction_path);
        cout << "Loaded " << reconstruction_path << endl;

        // Save original
        save_reconstruction(num_image_voxels_xyz, phantom_reconstruction, output_path);

        // cout << "\n <> Felzenszwalb Minimal Spanning Tree" << endl;
        // FelzenszwalbMST FMST = FelzenszwalbMST(num_image_voxels_xyz, phantom_reconstruction, output_path);
        // FMST.run(0, 0);
        // FMST.run(0, 0.005);
        // FMST.run(1, 0.005);

        cout << "\n <> Ganin Minimal Spanning Tree" << endl;
        GaninMST GMST = GaninMST(num_image_voxels_xyz, phantom_reconstruction, output_path);
        GMST.run();
    }
    else
    {
        cout << "Loading config and phantom files..." << endl;
        config = read_json_file(argv[1]);
        materials = read_json_file(argv[2]);

        image = ImageSpace();
        image.initializeFromConfig(&config, &materials);
        num_image_voxels_xyz = image.getNumImageVoxelsXYZ();
        phantom_materials = image.getPhantomMaterials();

        cout << "Loaded " << config["projection"]["phantom_file"] << endl;
        cout << "num_image_voxels_xyz = [" << num_image_voxels_xyz.x << ", " << num_image_voxels_xyz.y << ", " << num_image_voxels_xyz.z << "]" << endl;
        cout << "Total voxels: " << size(num_image_voxels_xyz) << endl;

        // Save original
        int x, y, z;
        x = num_image_voxels_xyz.x;
        y = num_image_voxels_xyz.y;
        z = num_image_voxels_xyz.z;
        string output_name = "OriginalPhantom_" + to_string(x) + "x" + to_string(y) + "x" + to_string(z) + "_8Bit" + ".raw";
        image.savePhantomMaterials(output_path, output_name);

        cout << "\n <> Simple Connecting Component Labelling:" << endl;
        SimpleCCL SCCL = SimpleCCL(num_image_voxels_xyz, phantom_materials, output_path);
        // 6 - connected
        SCCL.run(0);
        SCCL.saveSegmentedImage();
        // 18 - connected
        SCCL.run(1);
        SCCL.saveSegmentedImage();
        // 26-connected
        SCCL.run(2);
        SCCL.saveSegmentedImage();
        SCCL.clear();

        cout << "\n <> Playne-Equivalence (CUDA CCL):" << endl;
        PlayneEquivalenceCCL PeCCL = PlayneEquivalenceCCL(num_image_voxels_xyz, phantom_materials, output_path);
        PeCCL.run();
    }

    cout << "Finished" << endl;
}