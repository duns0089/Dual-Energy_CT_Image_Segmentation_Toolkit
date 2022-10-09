#pragma once
#include "Helpers.h"
#include <string>
#include <nlohmann/json.hpp>
using nlohmann::json;
using json_iter = nlohmann::detail::iteration_proxy_value<nlohmann::detail::iter_impl<nlohmann::json>>;
extern int trace_counter;

/**
* @class AttenuationCoefficient
* @brief Defines the voltage, mass attenuation coefficient pair for a MaterialCompound
* @see MaterialCompound
*/
class AttenuationCoefficient {
public:
	/**
	* @brief Constructs AttenuationCoefficient object with default values
	*/
	AttenuationCoefficient();

	/**
	* @brief Constructs AttenuationCoefficient object with a given energy and default mass attenuation coefficient
	* @param[in]	energy		voltage in keV for a voltage, mass attenuation coefficient pair
	*/
	AttenuationCoefficient(float energy);
	
	/**
	* @brief Constructs AttenuationCoefficient object with given energy and mass attenuation coefficient values
	* @param[in]	energy								voltage in keV for a voltage, mass attenuation coefficient pair
	* @param[in]	mass_attenuation_coefficient		mass_attenuation_coefficient in cm^2/g for a voltage, mass attenuation coefficient pair
	*/
	AttenuationCoefficient(float energy, float mass_attenuation_coefficient);

	/**
	* @brief Destroys AttenuationCoefficient object
	*/
	~AttenuationCoefficient();

	/**
	* @brief		Returns the voltage, in keV, of the voltage, mass attenuation coefficient pair
	* @return		voltage, in keV, of the voltage, mass attenuation coefficient pair
	*/
	float getEnergy();

	/**
	* @brief		Sets the voltage, in keV, for the voltage, mass attenuation coefficient pair
	* @param[in]	energy		voltage, in keV, of the voltage, mass attenuation coefficient pair
	*/
	void setEnergy(float energy);

	/**
	* @brief		Returns the mass attenuation coefficient, in cm^2/g, of the voltage, mass attenuation coefficient pair
	* @return		mass attenuation coefficient, in cm^2/g, of the voltage, mass attenuation coefficient pair
	*/
	float getMassAttenuationCoefficient();

	/**
	* @brief		Sets the mass attenuation coefficient, in cm^2/g, of the voltage, mass attenuation coefficient pair
	* @param[in]	mass_attenuation_coefficient		mass attenuation coefficient, in cm^2/g, of the voltage, mass attenuation coefficient pair
	*/
	void setMassAttenuationCoefficient(float mass_attenuation_coefficient);

	/**
	* @brief		Initializes, in CPU memory, the AttenuationCoefficient values provided in a material compounds JSON file
	* @param[in]	ac_iter			nlohmann::detail::iteration_proxy_value<nlohmann::detail::iter_impl<nlohmann::json>> object
	*								that corresponds to a JSON object describing the AttenuationCoefficient
	*/
	void initializeFromConfig(json_iter& ac_iter);

private:
	float energy; //!< in keV
	float mass_attenuation_coefficient; //!< mu/rho in cm2/g
};

/**
* @class MaterialCompound
* @brief Defines the properties of a material compound that will be imaged 
*/
class MaterialCompound
{
public:
	/**
	* @brief Constructs MaterialCompound object with default values
	*/
	MaterialCompound();

	/**
	* @brief Constructs MaterialCompound object with a given ID and default other values
	* @param[in]	id		Material Compound id, arbitrary to the material compound json file provided
	*/
	MaterialCompound(uint8_t id);

	/**
	* @brief Destroys MaterialCompound object
	*/
	~MaterialCompound();

	/**
	* @brief		Returns the unique ID number for the MaterialCompound
	* @return		unique ID number for the MaterialCompound
	*/
	uint8_t getId();

	/**
	* @brief		Sets the unique ID number for the MaterialCompound
	* @param[in]	id		unique ID number for the MaterialCompound
	*/
	void setId(uint8_t id);

	/**
	* @brief		Returns the name of the MaterialCompound
	* @return		name of the MaterialCompound
	*/
	std::string getName();

	/**
	* @brief		Sets the name of the MaterialCompound
	* @param[in]	name	name of the MaterialCompound
	*/
	void setName(std::string name);
	
	/**
	* @brief		Returns the density, in g/cm^3, of the MaterialCompound
	* @return		density (g/cm^3) of the MaterialCompound
	*/
	float getDensity();

	/**
	* @brief		Sets the density, in g/cm^3, of the MaterialCompound
	* @param[in]	density		density (g/cm^3) of the MaterialCompound
	*/
	void setDensity(float density);

	/**
	* @brief		Returns the number of AttenuationCoefficient for the MaterialCompound, provided by the json file
	* @return		number of AttenuationCoefficient for the MaterialCompound
	*/
	int getNumAttenuationCoefficients();

	/**
	* @brief		Sets the number of AttenuationCoefficient for the MaterialCompound, provided by the json file
	* @param[in]	num_attenuation_coefficient		the number of AttenuationCoefficient for the MaterialCompound
	*/
	void setNumAttenuationCoefficients(int num_attenuation_coefficients);

	/**
	* @brief		Returns the Pointer to the array of AttenuationCoefficients for the MaterialCompound
	* @return		Pointer to the array of AttenuationCoefficients for the MaterialCompound
	*/
	AttenuationCoefficient* getAttenuationCoefficients() const;

	/**
	* @brief		Initializes, in CPU memory, the MaterialCompound values provided in the JSON file, given an id number
	* @param[in]	material_json	Pointer to a nlohmann::json object that stores the data read from a materials config file
	* @param[in]	id				unique ID number for the MaterialCompound
	*/
	void initializeFromConfig(json* material_json, uint8_t id);

	/**
	* @brief		Finds or Calculates and Returns the attenuation value (1/mm) for a MaterialCompound, given a voltage (keV)
	* @param[in]	voltage			voltage in kV that we are using to determine the attenuation coefficient
	*								Since our Spectrum voltages are binned using a consistant bin size, this centered in the bin,
	*								so it is equal to half the bin size lower than the desired voltage
	* @param[in]	half_bin_size	Half of the bin size used for the Spectrum voltages. This allows us to get the desired voltage
	*								that is the upper limit of the voltage bin
	* @return		attenuation value, in 1/mm, for the MaterialCompound corresponding to a voltage in keV
	*/
	float getAttenuationAtVoltage(float voltage, float half_bin_size);

private:
	uint8_t id;	//!< Material Compound id, arbitrarily defined in the material compound json file
	std::string name; //!< Name of material compound
	float density; //!< Material density in g/cm3
	int num_attenuation_coefficients; //!< Number of Attenuation Coefficients for material in the material compound json file
	AttenuationCoefficient* attenuation_coefficients; //!< Pointer (array) of Attenuation Coefficients for material in cm2/g

	/**
	* @brief		Allocates the array of AttenuationCoefficients for the MaterialCompound with a size of num_attenuation_coefficients.
	*				All AttenuationCoefficients are created using the default constructor
	*/
	void mallocAttenuationCoefficients();
};

