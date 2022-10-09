#include "ImageSpace.h"

ImageSpace::ImageSpace()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    units = "mm";
    // Default values that quickly indicate uninitialized values
    num_image_voxels_xyz = make_int3(-99, -99, -99);
    voxel_pitch = make_float3(-99.0f, -99.0f, -99.0f);
    image_axis_length = make_float3(-99.0f, -99.0f, -99.0f);

    phantom_image = nullptr;
    phantom_materials = nullptr;
    backprojection = nullptr;
    reconstruction = nullptr;
    materials = nullptr;
    material_attenuation = nullptr;
    material_id = nullptr;
    scaling_factor = 1;

    // DEC_TRACE;
}
ImageSpace::~ImageSpace()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    delete[] phantom_image;
    delete[] phantom_materials;
    delete[] backprojection;
    delete[] reconstruction;
    delete[] materials;
    delete[] material_attenuation;
    delete[] material_id;
    // do not delete memory manager
    // DEC_TRACE;
}

void ImageSpace::saveImage(float *image, std::string path, std::string file_name)
{
    // SPDLOG_TRACE("FLOW: image={}, path={}, file_name={}", (void*)image, path, file_name);
    // INC_TRACE;

    // SPDLOG_INFO("Saving {}", file_name);
    // Use FILE functions to pass data

    // Define the specific file to open based on the view
    std::ostringstream filename;
    filename << path << "/" << file_name;

    std::string file_open;
    file_open = filename.str();

    // Open File and ensure it can be opened
    std::ofstream file(file_open, std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "ERROR: Can not open file: " << file_open.c_str() << std::endl;
        exit(1);
    }

    // Read data into buffer

    for (int height = 0; height < num_image_voxels_xyz.z; ++height)
    {
        for (int length = 0; length < num_image_voxels_xyz.y; ++length)
        {
            for (int width = 0; width < num_image_voxels_xyz.x; ++width)
            {
                int index = INDEX3D(height, width, length, num_image_voxels_xyz.z, num_image_voxels_xyz.x);
                file.write(reinterpret_cast<char *>(&image[index]), sizeof(float));
            }
        }
    }
    file.close();
    // DEC_TRACE;
}

int3 ImageSpace::getNumImageVoxelsXYZ() const
{
    return num_image_voxels_xyz;
}
void ImageSpace::setNumImageVoxelsXYZ(int3 num_image_voxels_xyz)
{
    this->num_image_voxels_xyz = num_image_voxels_xyz;
}

float3 ImageSpace::getVoxelPitch() const
{
    return voxel_pitch;
}
void ImageSpace::setVoxelPitch(float3 voxel_pitch)
{
    this->voxel_pitch = voxel_pitch;
}

float3 ImageSpace::getImageAxisLength() const
{
    return image_axis_length;
}
void ImageSpace::setImageAxisLength(float3 image_axis_length)
{
    this->image_axis_length = image_axis_length;
}

float3 ImageSpace::getOriginBoundaryVoxelXYZ() const
{
    return origin_boundary_voxel_xyz;
}
void ImageSpace::setOriginBoundaryVoxelXYZ(float3 origin_boundary_voxel_xyz)
{
    this->origin_boundary_voxel_xyz = origin_boundary_voxel_xyz;
}

float *ImageSpace::getPhantomImage()
{
    return phantom_image;
}
void ImageSpace::memsetPhantomImage()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    if (phantom_image != nullptr)
    {
        delete[] phantom_image;
    }

    phantom_image = new float[size(num_image_voxels_xyz)]();

    // DEC_TRACE;
}

uint8_t *ImageSpace::getPhantomMaterials()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    if (phantom_materials == nullptr)
    {
        printf("phantom_materials is nullptr\n");
        // DEC_TRACE;
        return nullptr;
    }
    // DEC_TRACE;
    return phantom_materials;
}
void ImageSpace::memsetPhantomMaterials()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;
    if (phantom_materials != nullptr)
    {
        // SPDLOG_DEBUG("phantom_materials is not nullptr");
        delete[] phantom_materials;
        // SPDLOG_DEBUG("Deleted non-nullptr phantom_materials");
    }
    phantom_materials = new uint8_t[size(num_image_voxels_xyz)]();
    // DEC_TRACE;
}

float *ImageSpace::getBackprojection()
{
    return backprojection;
}
void ImageSpace::memsetBackprojection()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;
    if (backprojection != nullptr)
    {
        delete[] backprojection;
    }
    backprojection = new float[size(num_image_voxels_xyz)]();
    // DEC_TRACE;
}

float *ImageSpace::getReconstruction()
{
    return reconstruction;
}
void ImageSpace::memsetReconstruction()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;
    if (reconstruction != nullptr)
    {
        delete[] reconstruction;
    }
    reconstruction = new float[size(num_image_voxels_xyz)]();
    // DEC_TRACE;
}

int ImageSpace::getScalingFactor() const
{
    return scaling_factor;
}
void ImageSpace::setScalingFactor(int scaling_factor)
{
    this->scaling_factor = scaling_factor;
}

int ImageSpace::getNumMaterials() const
{
    return num_materials;
}

void ImageSpace::setNumMaterials(int num_materials)
{
    this->num_materials = num_materials;
}

MaterialCompound *ImageSpace::getMaterials() const
{
    return materials;
}

MaterialCompound ImageSpace::getMaterialAtIndex(int material_index)
{
    return materials[material_index];
}

void ImageSpace::mallocMaterials()
{
    if (materials != nullptr)
    {
        delete[] materials;
    }
    materials = new MaterialCompound[num_materials]();
}

void ImageSpace::initializeFromConfig(json *config_json, json *material_json)
{
    // SPDLOG_TRACE("FLOW {}: config_json={}, material_json", trace_counter, (void*)config_json, (void*)material_json);
    // INC_TRACE;

    // SPDLOG_DEBUG("Reading image info");

    voxel_pitch = make_float3(
        (float)(*config_json)["image"]["pitch"]["x"],
        (float)(*config_json)["image"]["pitch"]["y"],
        (float)(*config_json)["image"]["pitch"]["z"]);
    num_image_voxels_xyz = make_int3(
        (int)(*config_json)["image"]["pixels"]["x"],
        (int)(*config_json)["image"]["pixels"]["y"],
        (int)(*config_json)["image"]["pixels"]["z"]);
    origin_boundary_voxel_xyz = 0.5f * (num_image_voxels_xyz - 1);
    image_axis_length = num_image_voxels_xyz * voxel_pitch;

    bool simulate_measurment = (bool)(*config_json)["projection"]["simulate_measurement"];
    // @note: take a look at upscaling here, and this if/else block
    if (simulate_measurment)
    { // Upscale and forward projection the phantom
        scaling_factor = (int)(*config_json)["projection"]["scaling_factor"];
        upscale();
    }
    else
    {
        memsetPhantomImage();
        memsetPhantomMaterials();
    }

    std::string path = (*config_json)["base_filepath"]; // Get base path
    std::string input_path = path + (std::string)(*config_json)["input_path"];
    std::string phantom_file_name = input_path + (std::string)(*config_json)["projection"]["phantom_file"];

    if (simulate_measurment)
    {
        readPhantomFromFile(phantom_file_name);
        flipPhantomIndices();
    }
    initializeMaterialsFromConfig(material_json);
    memsetReconstruction();

    // DEC_TRACE;
}

void ImageSpace::initializeMaterialsFromConfig(json *material_json)
{
    // SPDLOG_TRACE("FLOW {}: material_json={}", trace_counter, (void*)material_json);
    // INC_TRACE;

    if (phantom_materials == nullptr)
    {
        // SPDLOG_ERROR("PhantomMaterials is not initialized");
        exit(1);
    }

    std::set<uint8_t> material_id_set;
    for (int z = 0; z < num_image_voxels_xyz.z; ++z)
    {
        for (int x = 0; x < num_image_voxels_xyz.x; ++x)
        {
            for (int y = 0; y < num_image_voxels_xyz.y; ++y)
            {
                material_id_set.insert(phantom_materials[INDEX3D(z, x, y, num_image_voxels_xyz.z, num_image_voxels_xyz.x)]); // set will only take unique values
            }
        }
    }

    num_materials = (int)material_id_set.size();
    mallocMaterials();
    // SPDLOG_DEBUG("Created MaterialCompound array of size {}", material_id_set.size());
    int i = 0;
    for (std::set<uint8_t>::iterator id = material_id_set.begin(); id != material_id_set.end(); ++id)
    {
        materials[i].initializeFromConfig(material_json, *id);
        ++i;
    }

    // Verify Materials
    // SPDLOG_DEBUG("Materials Loaded from Config:");
    for (int j = 0; j < num_materials; ++j)
    {
        // SPDLOG_DEBUG("\tMaterial {} has id {}, is named {} has density {} g/cm3 with {} Attenuation Coefficients: ",
        //     j,
        //     materials[j].getId(),
        //     materials[j].getName(),
        //     materials[j].getDensity(),
        //     materials[j].getNumAttenuationCoefficients());
        for (int k = 0; k < materials[j].getNumAttenuationCoefficients(); ++k)
        {
            // SPDLOG_DEBUG("\t\tEnergy: {}\tkev,\tAttenuation: {} cm2/g",
            //     materials[j].getAttenuationCoefficients()[k].getEnergy(),
            //     materials[j].getAttenuationCoefficients()[k].getMassAttenuationCoefficient());
        }
    }

    material_attenuation = new float[num_materials];
    material_id = new uint8_t[num_materials];

    // DEC_TRACE;
}

void ImageSpace::readPhantomFromFile(std::string file_name)
{
    // SPDLOG_TRACE("FLOW {}: file_name={}", trace_counter, file_name);
    // INC_TRACE;

    // SPDLOG_DEBUG("Reading phantom");

    // Use FILE functions to pass data
    std::ifstream phantom_file(file_name, std::ios::binary);
    if (!phantom_file.is_open())
    {
        // SPDLOG_ERROR("Could not open file: {} ", file_name);
        // DEC_TRACE;
        exit(1);
    }
    // // Need to read phantom and check if size matches the info file
    // // Verify that the size from the json matches the phantom file size
    // if (fs::file_size(file_name.c_str()) != sizeof(uint8_t) * size(num_image_voxels_xyz))
    // {
    //     std::cout << "error 1 here" << std::endl;
    //     // SPDLOG_ERROR("Phantom buffer of size {} bytes does not equal the phantom file of size {} bytes", (sizeof(uint8_t) * size(num_image_voxels_xyz)), fs::file_size(file_name.c_str()));
    //     exit(1);
    // }

    // Read data into buffer
    if (phantom_materials == nullptr)
    {
        // SPDLOG_ERROR("Phantom Materials was not initialized and is nullptr");
        exit(1);
    }
    phantom_file.read(reinterpret_cast<char *>(phantom_materials), sizeof(uint8_t) * size(num_image_voxels_xyz));
    phantom_file.close();
    // DEC_TRACE;
}

void ImageSpace::upscale()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    voxel_pitch /= (float)scaling_factor;
    num_image_voxels_xyz *= scaling_factor;
    origin_boundary_voxel_xyz = 0.5f * (num_image_voxels_xyz - 1);
    // image_axis_length remains constant when upscaling
    memsetPhantomImage();
    memsetPhantomMaterials();

    // DEC_TRACE;
}

void ImageSpace::downscale()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    if (scaling_factor < 2)
    {
        // DEC_TRACE;
        return;
    }
    voxel_pitch *= (float)scaling_factor;
    num_image_voxels_xyz /= scaling_factor;
    origin_boundary_voxel_xyz = 0.5f * (num_image_voxels_xyz - 1);
    // image_axis_length remains constant when downscaling
    memsetPhantomImage();
    memsetPhantomMaterials();

    // DEC_TRACE;
}

void ImageSpace::flipPhantomIndices()
{
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    try
    {
        uint8_t *phantom_flipped = new uint8_t[size(num_image_voxels_xyz)];
        // SPDLOG_DEBUG("created phantom_flipped");
        for (int z = 0; z < num_image_voxels_xyz.z; ++z)
        {
            for (int y = 0; y < num_image_voxels_xyz.y; ++y)
            {
                for (int x = 0; x < num_image_voxels_xyz.x; ++x)
                {
                    int index = INDEX3D(x, y, z, num_image_voxels_xyz.x, num_image_voxels_xyz.y);
                    int index_flipped = INDEX3D(z, x, y, num_image_voxels_xyz.z, num_image_voxels_xyz.x);
                    phantom_flipped[index_flipped] = phantom_materials[index];
                }
            }
        }
        // SPDLOG_DEBUG("populated phantom_flipped");
        for (int i = 0; i < size(num_image_voxels_xyz); ++i)
        {
            phantom_materials[i] = phantom_flipped[i];
        }
        // SPDLOG_DEBUG("assigned phantom_flipped to phantom_materials");
        delete[] phantom_flipped;
        // SPDLOG_DEBUG("end flipPhantomIndices");
    }
    catch (std::exception &e)
    {
        // SPDLOG_ERROR("Flipping Phantom Indices failed, here is why: {}", e.what());
    }
    // DEC_TRACE;
}

void ImageSpace::savePhantom(std::string path, std::string file_name)
{
    // SPDLOG_TRACE("FLOW: path={}, file_name={}", path, file_name);
    // INC_TRACE;
    saveImage(phantom_image, path, file_name);
    // DEC_TRACE;
}

void ImageSpace::saveReconstruction(std::string path, std::string file_name)
{
    // SPDLOG_TRACE("FLOW: path={}, file_name={}", path, file_name);
    // INC_TRACE;
    saveImage(reconstruction, path, file_name);
    // DEC_TRACE;
}

void ImageSpace::saveBackprojection(std::string path, std::string file_name)
{
    // SPDLOG_TRACE("FLOW: path={}, file_name={}", path, file_name);
    // INC_TRACE;
    saveImage(backprojection, path, file_name);
    // DEC_TRACE;
}

void ImageSpace::savePhantomMaterials(std::string path, std::string file_name)
{
    // SPDLOG_TRACE("FLOW {}: path={}, file_name={}", trace_counter, path, file_name);
    // INC_TRACE;

    // SPDLOG_INFO("Saving {}", file_name);
    // Use FILE functions to pass data

    // Define the specific file to open based on the view
    std::ostringstream filename;
    filename << path << "/" << file_name;

    std::string file_open;
    file_open = filename.str();

    // Open File and ensure it can be opened
    std::ofstream file(file_open, std::ios::binary);
    if (!file.is_open())
    {
        // SPDLOG_ERROR("ERROR: Can not open file: {}", file_open.c_str());
        exit(1);
    }

    // Read data into buffer

    for (int height = 0; height < num_image_voxels_xyz.z; ++height)
    {
        for (int length = 0; length < num_image_voxels_xyz.y; ++length)
        {
            for (int width = 0; width < num_image_voxels_xyz.x; ++width)
            {
                int index = INDEX3D(height, width, length, num_image_voxels_xyz.z, num_image_voxels_xyz.x);
                file.write(reinterpret_cast<char *>(&phantom_materials[index]), sizeof(uint8_t));
            }
        }
    }
    file.close();
    // DEC_TRACE;
}
