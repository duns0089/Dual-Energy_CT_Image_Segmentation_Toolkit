#pragma once
#include "Helpers.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
// #include <nlohmann/json-schema.hpp>
#include <set>

#include "MaterialCompound.h"
using nlohmann::json;
namespace fs = std::filesystem;
extern int trace_counter;

/**
* @class ImageSpace
* @brief Defines the volume being reconstructed, including dimensions, phantom, materials, and reconstruction data
*/
class ImageSpace
{
private:
	std::string			units;						//!< Default units are mm unless overridden in config json
	int3				num_image_voxels_xyz;		//!< Number of voxels in the image in x,y,z
	float3				voxel_pitch;				//!< Image voxel pitch on grid
	float3				image_axis_length;			//!< Grid space size of image
	float3				origin_boundary_voxel_xyz;	//!< Number of voxels from the image origin to the center of the boundary voxel
	float*				phantom_image;				//!< Phantom Image attenuation data in a 1D array. Index read as z,x,y = z + (x*z_length) + (y*z_length*x_length)
	uint8_t*			phantom_materials;			//!< Phantom Image Material Compound Id data in a 1D array. Index read as x,y,z = x + (y*x_length) + (z*x_length*y_length) -> then index is flipped to z,x,y for optimal memory access
	float*				backprojection;				//!< Backprojection Image data in a 1D array
	float*				reconstruction;             //!< Reconstruction Image data in a 1D array, where index is z,x,y for optimal memory access
	int					scaling_factor;				//!< How much to upscale the system for simulating a measurement

	int					num_materials;				//!< Number of distinct Material Compounds present in the phantom
	MaterialCompound*	materials;					//!< The full set of Material Compounds present in the phantom. Will be accessed when setting the phantom attenuation values for given spectrum energy

	float*				material_attenuation;		//!< Array of the attenuation values for the Material Compounds present in the phantom
	uint8_t*			material_id;				//!< Array of the id values for the Material Compounds present in the phantom

	std::string			phantom_image_key = "reconstruction";				//!< Symbol key for the phantom image array living in the GPU memory map
	std::string			phantom_materials_key = "phantom_materials";		//!< Symbol key for the phantom materials array living in the GPU memory map
	std::string			material_attenuation_key = "material_attenuation";	//!< Symbol key for the material attenuation array living in the GPU memory map
	std::string			material_id_key = "material_id";					//!< Symbol key for the material id array living in the GPU memory map

	/**
	* @brief		Saves the Image values to a RAW file on disk. The image to be saved is chosen by a public wrapper function
	* @see			ImageSpace::savePhantom(std::string path, std::string file_name)
	* @see			ImageSpace::saveReconstruction(std::string path, std::string file_name);
	* @param[in]	image		Pointer to array holding the values of the image being saved
	* @param[in]	path		Folder location where the image should be saved
	* @param[in]	file_name	The name of the file to be written, including the suffix extension (.raw)
	*/
	void saveImage(float* image, std::string path, std::string file_name);

	/**
	* @brief		Reads a phantom image from a file on disk and stores the data into the appropriate array
	* @param[in]	file_name	The file name of the file from which the data should be read
	*/
	void readPhantomFromFile(std::string file_name);

public:
	/**
	* @brief Constructs ImageSpace object with default values
	*/
	ImageSpace();

	/**
	* @brief Destroys ImageSpace object
	*/
	~ImageSpace();

	/**
	* @brief		Returns the number of voxels, in 3 dimensions, present in the ImageSpace
	* @return		number of voxels in 3 dimensions (x,y,z)
	*/
	int3 getNumImageVoxelsXYZ() const;

	/**
	* @brief		Sets the number of voxels, in 3 dimensions, present in the Image Space
	* @param[in]	num_image_voxels_xyz	number of voxels in 3 dimensions (x,y,z)
	*/
	void setNumImageVoxelsXYZ(int3 num_image_voxels_xyz);

	/**
	* @brief		Returns the voxel pitch (or size of a single voxel), in 3 dimensions, for the ImageSpace
	* @return		voxel pitch in 3 dimensions (x,y,z) in millimeters
	*/
	float3 getVoxelPitch() const;

	/**
	* @brief		Sets the voxel pitch (or size of a single voxel), in 3 dimensions, for the ImageSpace
	* @param[in]	voxel_pitch		voxel pitch in 3 dimensions (x,y,z) in millimeters
	*/
	void setVoxelPitch(float3 voxel_pitch);

	/**
	* @brief		Returns the full size of the ImageSpace, in 3 dimensions
	* @return		size of ImageSpace in 3 dimensions (x,y,z) in millimeters
	*/
	float3 getImageAxisLength() const;

	/**
	* @brief		Sets the full size of the ImageSpace, in 3 dimensions
	* @param[in]	image_axis_length	size of ImageSpace in 3 dimensions (x,y,z) in millimeters
	*/
	void setImageAxisLength(float3 image_axis_length);

	/**
	* @brief		Returns the distance from the center of ImageSpace to edge, in 3 dimensions
	* @return		Distance from center of ImageSpace to the center of the edge voxel
	*				in 3 dimensions (x,y,z), in number of voxels
	*/
	float3 getOriginBoundaryVoxelXYZ() const;
	
	/**
	* @brief		Sets the distance from the center of ImageSpace to edge, in 3 dimensions
	* @param[in]	origin_boundary_voxel_xyz	Distance from the center of ImageSpace to
	*											the center of edge voxel in 3 dimensions (x,y,z),
	*											in number of voxels
	*/
	void setOriginBoundaryVoxelXYZ(float3 origin_boundary_voxel_xyz);

	/**
	* @brief		Returns the pointer to an array storing the x-ray attenuation values of the phantom
	* @return		Pointer to the array storing the x-ray attenuation values of the phantom
	*/
	float* getPhantomImage();

	/**
	* @brief		Allocates the array of the x-ray attenuation values of the phantom. Array is of 
	*				size num_image_voxels_xyz All values are set to 0.0f
	*/
	void memsetPhantomImage();

	/**
	* @brief		Returns the pointer to an array storing the material id numbers of the phantom
	* @return		Pointer to the array storing the material id numbers of the phantom
	*/
	uint8_t* getPhantomMaterials();

	float* getBackprojection();
	void memsetBackprojection();

	/**
	* @brief		Allocates the array of the material id numbers of the phantom. Array is of
	*				size num_image_voxels_xyz All values are set to 0
	*/
	void memsetPhantomMaterials();

	/**
	* @brief		Returns the pointer to an array storing the x-ray attenuation values of the reconstruction
	* @return		Pointer to the array storing the x-ray attenuation values of the reconstruction
	*/
	float* getReconstruction();

	/**
	* @brief		Allocates the array of the x-ray attenuation values of the reconstruction. Array is of
	*				size num_image_voxels_xyz All values are set to 0.0f
	*/
	void memsetReconstruction();

	/**
	* @brief		Returns the scaling factor to adjust the size of the ImageSpace
	* @return		Scaling Factor to adjust the size of the ImageSpace
	*/
	int getScalingFactor() const;

	/**
	* @brief		Sets scaling factor to adjust the size of the ImageSpace
	* @param[in]	scaling_factor	scaling factor to adjust the size of the ImageSpace
	*/
	void setScalingFactor(int scaling_factor);

	/**
	* @brief		Returns the number of materials present in the phantom
	* @return		Number of materials present in the phantom
	*/
	int getNumMaterials() const;

	/**
	* @brief		Sets number of materials present in the phantom
	* @param[in]	num_materials	Number of materials present in the phantom
	*/
	void setNumMaterials(int num_materials);

	/**
	* @brief		Returns the array of MaterialCompounds present in the phantom
	* @return		Pointer to array of MaterialCompounds present in the phantom
	*/
	MaterialCompound* getMaterials() const;

	/**
	* @brief		Returns a MaterialCompound indexed in the array of MaterialCompounds present in the phantom
	* @param[in]	material_index	The index of the desired MaterialCompound
	* @return		MaterialCompound at a given index in the array of MaterialCompounds present in the phantom
	*/
	MaterialCompound getMaterialAtIndex(int material_index);
	
	/**
	* @brief		Allocates the array of MaterialCompounds in the phantom with a size of num_materials. All MaterialCompounds are created using the default constructor
	*/
	void mallocMaterials();

	/**
	* @brief		Initializes, in CPU memory, the ImageSpace values provided in JSON files
	* @param[in]	config_json		Pointer to a nlohmann::json object that stores the data read from a system config file
	* @param[in]	material_json	Pointer to a nlohmann::json object that stores the data read from a materials config file
	*/
	void initializeFromConfig(json* config_json, json* material_json);

	/**
	* @brief		Initializes, in CPU memory, the ImageSpace MaterialCompound values provided in the JSON file
	* @param[in]	material_json	Pointer to a nlohmann::json object that stores the data read from a materials config file
	*/
	void initializeMaterialsFromConfig(json* material_json);

	/**
	* @brief		Increases the size if the ImageSpace and phantom by a scaling factor
	*/
	void upscale();

	/**
	* @brief		Decreases the size if the ImageSpace and phantom by a scaling factor
	*/
	void downscale();

	/**
	* @brief		Flips the indexing of the phantom image from (x,y,z) to (z,x,y).
	*				This is done for optimal memory read and write speeds
	*/
	void flipPhantomIndices();

	/**
	* @brief		Saves the Phantom Image attenuation values to a RAW file on disk
	* @param[in]	path		Folder location where the image should be saved
	* @param[in]	file_name	The name of the file to be written, including the suffix extension (.raw)
	*/
	void savePhantom(std::string path, std::string file_name);

	/**
	* @brief		Saves the reconstruction attenuation values to a RAW file on disk
	* @param[in]	path		Folder location where the image should be saved
	* @param[in]	file_name	The name of the file to be written, including the suffix extension (.raw)
	*/
	void saveReconstruction(std::string path, std::string file_name);

	void saveBackprojection(std::string path, std::string file_name);

	/**
	* @brief		Saves the Phantom Image Material ID values to a RAW file on disk
	* @param[in]	path		Folder location where the image should be saved
	* @param[in]	file_name	The name of the file to be written, including the suffix extension (.raw)
	*/
	void savePhantomMaterials(std::string path, std::string file_name);

	/**
	* @brief		Sets the attenuation values for a phantom. Attenuation is voltage dependent
	* @param[in]	voltage			voltage in kV that we are using to determine the attenuation for all materials in the phantom.
	*								Since our Spectrum voltages are binned using a consistant bin size, this centered in the bin,
	*								so it is equal to half the bin size lower than the desired voltage
	* @param[in]	half_bin_size	Half of the bin size used for the Spectrum voltages. This allows us to get the desired voltage
	*								that is the upper limit of the voltage bin
	*/
	void setPhantomAttenuation(float voltage, float half_bin_size);

	/**
	* @brief		Returns the Pointer of the array of the Phantom attenuation values stored in GPU memory
	* @return		The array of the Phantom attenuation values stored in GPU memory
	*/
	float*	getDevicePointer_PhantomImage();

	/**
	* @brief		Returns the dimensions of the allocated GPU memory for the Phantom attenuation values
	* @return		Dimensions of the allocated GPU memory. Dimensions are 3D (x,y,z) plus w (pitch)
	*/
	int4	getDeviceDim_PhantomImage();

	/**
	* @brief		Sets the values of the array of Phantom attenuation to 0.0f in the memory on the GPU device
	*/
	void	memsetDevice_PhantomImage();

	/**
	* @brief		Copies the Phantom attenuation values from CPU host memory to GPU device memory
	*/
	void	copyHostToDevice_PhantomImage();

	/**
	* @brief		Copies the Phantom attenuation values from GPU device memory to CPU host memory
	*/
	void    copyDeviceToHost_PhantomImage();

	/**
	* @brief		Returns the Pointer of the array of the reconstruction attenuation values stored in GPU memory
	* @return		The array of the Phantom attenuation values stored in GPU memory
	*/
	float*	getDevicePointer_Reconstruction();

	/**
	* @brief		Returns the dimensions of the allocated GPU memory for the reconstruction attenuation values
	* @return		Dimensions of the allocated GPU memory. Dimensions are 3D (x,y,z) plus w (pitch)
	*/
	int4	getDeviceDim_Reconstruction();

	/**
	* @brief		Sets the values of the array of reconstruction attenuation to 0.0f in the memory on the GPU device
	*/
	void	memsetDevice_Reconstruction();

	/**
	* @brief		Copies the reconstruction attenuation values from CPU host memory to GPU device memory
	*/
	void	copyHostToDevice_Reconstruction();

	/**
	* @brief		Copies the reconstruction attenuation values from GPU device memory to CPU host memory
	*/
	void    copyDeviceToHost_Reconstruction();



};

