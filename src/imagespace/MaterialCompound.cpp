#include "MaterialCompound.h"

AttenuationCoefficient::AttenuationCoefficient() {
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    energy = -1.0f;
    mass_attenuation_coefficient = -1.0f;
}

AttenuationCoefficient::AttenuationCoefficient(float energy) {
    // SPDLOG_TRACE("FLOW {}: energy={}", trace_counter, energy);
    this->energy = energy;
    mass_attenuation_coefficient = -1.0f;
}

AttenuationCoefficient::AttenuationCoefficient(float energy, float mass_attenuation_coefficient) {
    // SPDLOG_TRACE("FLOW {}: energy={}, mass_attenuation_coefficient={}", trace_counter, energy, mass_attenuation_coefficient);
    this->energy = energy;
    this->mass_attenuation_coefficient = mass_attenuation_coefficient;
}

AttenuationCoefficient::~AttenuationCoefficient() {
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
}

float AttenuationCoefficient::getEnergy() {
    return energy;
}

void AttenuationCoefficient::setEnergy(float energy) {
    this->energy = energy;
}

float AttenuationCoefficient::getMassAttenuationCoefficient() {
    return mass_attenuation_coefficient;
}

void AttenuationCoefficient::setMassAttenuationCoefficient(float mass_attenuation_coefficient) {
    this->mass_attenuation_coefficient = mass_attenuation_coefficient;
}

void AttenuationCoefficient::initializeFromConfig(json_iter& ac_iter) {
    // SPDLOG_TRACE("FLOW {}: ac_iter={}", trace_counter, (void*)&ac_iter);
    // INC_TRACE;
    energy = (float)ac_iter.value()["energy"] * MEV_TO_KEV;
    mass_attenuation_coefficient = (float)ac_iter.value()["up"];
    // DEC_TRACE;
}

MaterialCompound::MaterialCompound() {
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    id = -1;
    name = "";
    density = -1.0f;
    num_attenuation_coefficients = -1;
    attenuation_coefficients = nullptr;

    // DEC_TRACE;
}

MaterialCompound::MaterialCompound(uint8_t id) {
    // SPDLOG_TRACE("FLOW {}: id={}", trace_counter, id);
    // INC_TRACE;
    this->id = id;
    name = "";
    density = -1.0f;
    num_attenuation_coefficients = -1;
    attenuation_coefficients = nullptr;
    // DEC_TRACE;
}

MaterialCompound::~MaterialCompound() {
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;
    delete[] attenuation_coefficients;
    // DEC_TRACE;
}

uint8_t MaterialCompound::getId() {
    return id;
}

void MaterialCompound::setId(uint8_t id) {
    this->id = id;
}

std::string MaterialCompound::getName() {
    return name;
}

void MaterialCompound::setName(std::string name) {
    this->name = name;
}

float MaterialCompound::getDensity() {
    return density;
}

void MaterialCompound::setDensity(float density) {
    this->density = density;
}

int MaterialCompound::getNumAttenuationCoefficients() {
    return num_attenuation_coefficients;
}

void MaterialCompound::setNumAttenuationCoefficients(int num_attenuation_coefficients) {
    this->num_attenuation_coefficients = num_attenuation_coefficients;
}

AttenuationCoefficient* MaterialCompound::getAttenuationCoefficients() const{
    return attenuation_coefficients;
}

void MaterialCompound::initializeFromConfig(json* material_json, uint8_t id) {
    // SPDLOG_TRACE("FLOW {}: material_json={}, id={}", trace_counter, (void*)material_json, id);
    // INC_TRACE;
    this->id = id;
    for (json_iter& compound : (*material_json)["compounds"].items()) {
        if (compound.value()["Compound_Id"] == id) {
            name = compound.value()["Compound"];
            density = (float)compound.value()["Density"]; //@note not grabing density values properly. Why?
            // SPDLOG_DEBUG("{} has density {} and read {} from config", name, density, (float)compound.value()["Density"]);
            //exit(1);
            num_attenuation_coefficients = (int)compound.value()["Attenuation"].size();
            mallocAttenuationCoefficients();
            // SPDLOG_DEBUG("MaterialCompound: {} has {} AttenuationCoefficients", name, num_attenuation_coefficients);
            int i = 0;
            for (json_iter& ac : compound.value()["Attenuation"].items()) {
                attenuation_coefficients[i].initializeFromConfig(ac);
                ++i;
            }

            // DEC_TRACE;
            return;
        }
    }

    // SPDLOG_ERROR("Material with Id {} was not found in the JSON", id);
    exit(1);
}

float MaterialCompound::getAttenuationAtVoltage(float voltage, float half_bin_size) {
    // SPDLOG_TRACE("FLOW {}: voltage={}, half_bin_size={}", trace_counter, voltage, half_bin_size);
    // INC_TRACE;
    float voltage_lower = voltage - half_bin_size;
    float voltage_upper = voltage + half_bin_size;

    if (num_attenuation_coefficients > 0) {
        AttenuationCoefficient upperbound = AttenuationCoefficient(HUGE_VALF);
        AttenuationCoefficient lowerbound = AttenuationCoefficient(0.0f);
        for (int i = 0; i < num_attenuation_coefficients; ++i) {
            float material_energy = attenuation_coefficients[i].getEnergy();
            float material_attenuation = attenuation_coefficients[i].getMassAttenuationCoefficient() * density * MM_TO_CM; // (cm2/g)*(g/cm3)*(cm/mm) = 1/mm
            // if inside energy bin
            if (material_energy >= voltage_lower && material_energy <= voltage_upper) {
                // SPDLOG_DEBUG("Exact Energy Match Found. Returning {} mm^-1 for {} at {} keV", material_attenuation, name, voltage);
                // DEC_TRACE;
                return material_attenuation;
            }
            else if (material_energy < voltage_lower) {

                lowerbound.setEnergy(fmaxf(lowerbound.getEnergy(), material_energy));
                if (lowerbound.getEnergy() == material_energy) {
                    lowerbound.setMassAttenuationCoefficient(material_attenuation);
                }

            }
            else if (material_energy > voltage_upper) {
                upperbound.setEnergy(fminf(upperbound.getEnergy(), material_energy));
                if (upperbound.getEnergy() == material_energy) {
                    upperbound.setMassAttenuationCoefficient(material_attenuation);
                }
            }
            else {
                // SPDLOG_ERROR("Should not be here. Here are the current values: voltage_lower={}, volatge_upper={}, lowerbound.energy={}, upperbound.energy={}",
                //     voltage_lower, voltage_upper, lowerbound.getEnergy(), upperbound.getEnergy());
                exit(1);
            }
        }
        //// SPDLOG_DEBUG("\tLowerBound: (Energy {}, AC: {}),  Upperbound: (Energy {}, AC: {}), Voltage {}", lowerbound.getEnergy(), lowerbound.getMassAttenuationCoefficient(), upperbound.getEnergy(), upperbound.getMassAttenuationCoefficient(),  voltage);
        float attenuation = lerp(lowerbound.getMassAttenuationCoefficient(), upperbound.getMassAttenuationCoefficient(), -(voltage - lowerbound.getEnergy()) / (lowerbound.getEnergy() - upperbound.getEnergy()));
        // SPDLOG_DEBUG("Energy exact match not found. Interpolating attenuation value {} mm^-1 for {} at {} keV", attenuation, name, voltage);
        // DEC_TRACE;
        return attenuation;
    }

    // SPDLOG_ERROR("No Attenuation Coefficients Found for Material {} at voltage {} keV", name, voltage);
    exit(1);
}

void MaterialCompound::mallocAttenuationCoefficients() {
    // SPDLOG_TRACE("FLOW {}: ()", trace_counter);
    // INC_TRACE;

    if (attenuation_coefficients != nullptr) {
        delete[] attenuation_coefficients;
    }
    attenuation_coefficients = new AttenuationCoefficient[num_attenuation_coefficients]();
    // DEC_TRACE;
}
