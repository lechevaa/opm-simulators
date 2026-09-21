// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2025 NORCE Research AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HYBRID_NEWTON_HPP
#define HYBRID_NEWTON_HPP


#include <opm/simulators/flow/FlowBaseProblemProperties.hpp>
#include <opm/simulators/flow/HybridNewtonConfig.hpp>
#include <opm/simulators/flow/HybridNewtonReservoirInit.hpp>
#include <opm/input/eclipse/Units/UnitSystem.hpp>
#include <opm/models/blackoil/blackoilnewtonmethodparams.hpp>

#include <opm/ml/ml_model.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace Opm {

/*!
 * \brief Hybrid Newton solver extension for the black-oil model.
 *
 * This class integrates machine learning–based corrections into the
 * standard Newton solver of the black-oil simulator. It uses one or
 * more trained Hybrid Newton models to adjust the initial guess of the
 * nonlinear solver at specific timesteps.
 *
 * Each model is described by a HybridNewtonConfig, which specifies:
 *   - the path to the model file,
 *   - the cells to which the model applies,
 *   - the time points when it should be applied,
 *   - the input and output features with their transformations and scalings.
 *
 * At runtime, the class:
 *   1. Loads the relevant configuration(s),
 *   2. Constructs an input tensor [n_cells x n_input_features],
 *   3. Runs the ML model to produce an output tensor [n_cells x n_output_features],
 *   4. Updates the nonlinear solver’s initial guess accordingly.
 */
template <typename TypeTag>
class BlackOilHybridNewton
{
protected:
    using Simulator  = GetPropType<TypeTag, Properties::Simulator>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using MaterialLaw = GetPropType<TypeTag, Properties::MaterialLaw>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using Scalar  = GetPropType<TypeTag, Properties::Scalar>;

    enum { numPhases = FluidSystem::numPhases };

    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };
    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };

    static constexpr bool compositionSwitchEnabled = Indices::compositionSwitchIdx >= 0;

public:
    explicit BlackOilHybridNewton(Simulator& simulator)
        : simulator_(simulator)
        , reservoirInit_(simulator)
        , configsLoaded_(false)
    {}

    /*!
    * \brief Attempt to apply the Hybrid Newton correction at the current timestep.
    *
    * This function acts as the entry point for Hybrid Newton corrections.
    * It first checks whether the Hybrid Newton mechanism is enabled
    * (via `Parameters::UseHybridNewton`). If enabled, it validates the fluid system
    * and loads model configurations from the parameter-specified
    * configuration file (if not already loaded).
    *
    * At each timestep, it iterates over all loaded configurations and applies
    * those that match the current simulation time.
    *
    * Typical call site: at the beginning of a nonlinear solve for each timestep.
    *
    * \throws std::runtime_error if configuration validation fails or if an
    *         invalid fluid system is encountered.
    */
    void tryApplyHybridNewton()
    {
        tryApplyReservoirHybridNewton();
    }

    void tryApplyReservoirHybridNewton()
    {
        if (!Parameters::Get<Parameters::UseHybridNewton>())
            return;

        validateFluidSystem();

        loadConfigsIfNeeded();

        Scalar current_time = simulator_.time();
        for (const auto& config : configs_) {
            const bool should_apply = shouldApplyHybridNewton(current_time, config);
            if (should_apply) {
                runReservoirHybridNewton(config);
            }
        }
    }

private:
    void loadConfigsIfNeeded()
    {
        if (configsLoaded_) {
            return;
        }

        std::string config_file = Parameters::Get<Parameters::HybridNewtonConfigFile>();
        PropertyTree pt(config_file);
        for (const auto& model_key : pt.get_child_keys()) {
            HybridNewtonConfig config(pt.get_child(model_key));
            config.validateConfig(compositionSwitchEnabled);
            configs_.push_back(std::move(config));
        }
        configsLoaded_ = true;
    }

    /*!
    * \brief Apply the Hybrid Newton method using a given model configuration.
    *
    * This function loads the cell indices on demand from the configuration's
    * `cell_indices_file`, constructs the input tensor, evaluates the model to
    * produce the output tensor, and applies the result to update the initial
    * guess of the nonlinear solver.
    *
    * \param config The HybridNewtonConfig object containing model path,
    *        feature specifications, and cell indices file reference.
    *
    * \throws std::runtime_error if cell indices cannot be loaded or if
    *         feature extraction or application fails.
    */
    void runReservoirHybridNewton(const HybridNewtonConfig& config)
    {
        reservoirInit_.runReservoirHybridNewton(
            config,
            std::bind(&BlackOilHybridNewton::constructInputTensor, this, std::placeholders::_1),
            std::bind(&BlackOilHybridNewton::constructOutputTensor, this, std::placeholders::_1, std::placeholders::_2),
            [](const HybridNewtonConfig& current_config,
               const ML::Tensor<Evaluation>& input,
               const ML::Tensor<Evaluation>& output) {
                (void)current_config;
                (void)input;
                (void)output;
            });
    }

protected:
    void validateFluidSystem()
    {
        const auto& eclState = simulator_.vanguard().eclState();
        const auto& phases = eclState.runspec().phases();

        bool hasWater = phases.active(Phase::WATER);
        bool hasGas = phases.active(Phase::GAS);
        bool hasOil = phases.active(Phase::OIL);
        bool hasSolvent = phases.active(Phase::SOLVENT);

        // Check for three-phase black oil system
        if (!(hasWater && hasOil && hasGas && !hasSolvent)) {
            OPM_THROW(std::runtime_error,
                "HybridNewton: Unsupported fluid system. Only three-phase black oil is supported.");
        }
    }

    /*!
    * \brief Check whether the Hybrid Newton method should be applied at the given time.
    *
    * This function evaluates the current simulation time against the application times
    * specified in the HybridNewtonConfig. The behavior depends on how many entries are
    * provided in `config.apply_times`:
    *
    * - If one entry is provided, the method is applied exactly at that time (within
    *   a tolerance of 1e-6).
    * - If two entries are provided, the method is applied during the inclusive interval
    *   between the start and end times.
    * - For any other number of entries, the method is not applied.
    *
    * \param current_time The current simulation time.
    * \param config The HybridNewtonConfig specifying when the Hybrid Newton method
    *        should be applied.
    *
    * \return True if the Hybrid Newton method should be applied at \p current_time,
    *         false otherwise.
    */
    bool shouldApplyHybridNewton(Scalar current_time, const HybridNewtonConfig& config) const
    {
        constexpr Scalar tolerance = 1e-6;
        const Scalar current_dt = simulator_.timeStepSize();
        const Scalar current_end_time = current_time + current_dt;

        if (!std::isnan(config.apply_dt) &&
            std::abs(current_dt - config.apply_dt) >= tolerance) {
            return false;
        }

        if (!std::isnan(config.apply_start_time) &&
            std::abs(current_time - config.apply_start_time) >= tolerance) {
            return false;
        }

        if (!std::isnan(config.apply_end_time) &&
            std::abs(current_end_time - config.apply_end_time) >= tolerance) {
            return false;
        }

        if (config.apply_report_step >= 0 &&
            simulator_.episodeIndex() != config.apply_report_step) {
            return false;
        }

        if (config.apply_current_step >= 0 &&
            simulator_.timeStepIndex() != config.apply_current_step) {
            return false;
        }

        if (config.apply_times.size() == 1) {
            // Apply exactly at one point in time (with optional tolerance)
            bool apply = std::abs(current_time - config.apply_times[0]) < tolerance;
            return apply;
        }

        if (config.apply_times.size() == 2) {
            Scalar start_time = config.apply_times[0];
            Scalar end_time = config.apply_times[1];
            bool apply = (current_time >= start_time) && (current_time <= end_time);
            return apply;
        }

        // If apply_times has an unexpected number of entries
        return false;
    }


    /*!
    * \brief Construct the input feature tensor for the Hybrid Newton model.
    *
    * The tensor is constructed in the exact order specified by
    * `config.input_features`:
    *   - Scalar features (e.g., TIMESTEP) contribute one entry each.
    *   - Per-cell features (e.g., PRESSURE, SGAS) contribute a block of
    *     `n_cells` entries, one per cell index in `config.cell_indices`.
    *
    * Example:
    *   Suppose `config.input_features = [SGAS, TIMESTEP, PRESSURE]`
    *   and `config.n_cells = 3`. The resulting tensor layout is:
    *
    *     [ SGAS(cell0), SGAS(cell1), SGAS(cell2),
    *       TIMESTEP,
    *       PRESSURE(cell0), PRESSURE(cell1), PRESSURE(cell2) ]
    *
    * Thus, the tensor length equals:
    *
    *     (# of scalar features) + (# of per-cell features × n_cells).
    *
    * \param config  HybridNewtonConfig containing feature definitions and cell indices.
    * \return        A 1D tensor of Evaluation values with the computed layout.
    */
    ML::Tensor<Evaluation>
    constructInputTensor(const HybridNewtonConfig& config)
    {
        const auto& features = config.input_features;

        // compute total length directly
        std::size_t input_tensor_length = 0;
        for (const auto& feature : features) {
            const FeatureSpec& spec = feature.second;
            if (spec.actual_name == "TIMESTEP") {
                input_tensor_length += 1;
            } else {
                input_tensor_length += config.n_cells;
            }
        }

        ML::Tensor<Evaluation> input(input_tensor_length);
        std::size_t offset = 0;

        // fill in exact feature order from config
        for (const auto& feature: features) {
            const FeatureSpec& spec = feature.second;

            if (spec.actual_name == "TIMESTEP") {
                auto value = static_cast<Evaluation>(getScalarFeatureValue(spec));
                input(offset++) = value;
            } else {
                for (std::size_t i = 0; i < config.n_cells; ++i) {
                    auto value = static_cast<Evaluation>(
                        getPerCellFeatureValue(spec, config.cell_indices[i])
                    );
                    input(offset + i) = value;
                }
                offset += config.n_cells;
            }
        }

        return input;
    }

    /*!
    * \brief Retrieve and transform a scalar feature (global across the domain).
    *
    * Supported scalar feature:
    *   - "TIMESTEP": current timestep size.
    *
    * The raw value is passed through the feature's transformation
    * and scaling functions before being returned.
    *
    * \param spec The feature specification, including transform and scaling.
    * \return The transformed and scaled feature value.
    *
    * \throws std::runtime_error if the feature is unknown.
    */
    Scalar getScalarFeatureValue(const FeatureSpec& spec)
    {
        Scalar value = 0.0;

        if (spec.actual_name == "TIMESTEP") {
            value = simulator_.timeStepSize();
        } else {
            OPM_THROW(std::runtime_error, "Unknown scalar feature: " + spec.actual_name);
        }

        value = spec.transform.apply(value);
        return spec.scaler.scale(value);
    }

    /*!
    * \brief Retrieve and transform a per-cell feature value.
    *
    * Supported per-cell features include:
    *   - PRESSURE, SWAT, SOIL, SGAS, RS, RV, PERMX.
    *
    * The raw value is taken from the simulator state,
    * converted into the configured unit system,
    * then passed through the feature's transformation
    * and scaling functions.
    *
    * \param spec The feature specification, including transform and scaling.
    * \param cell_index The cell index for which to retrieve the value.
    * \return The transformed and scaled feature value.
    *
    * \throws std::runtime_error if the feature is unknown.
    */
    Scalar getPerCellFeatureValue(const FeatureSpec& spec, int cell_index)
    {
        const auto& intQuants = simulator_.model().intensiveQuantities(cell_index, 0);
        const auto& fs = intQuants.fluidState();
        const auto& unitSyst = simulator_.vanguard().schedule().getUnits();

        Scalar value = 0.0;

        if (spec.actual_name == "PRESSURE") {
            value = getValue(fs.pressure(oilPhaseIdx));
            value = unitSyst.from_si(UnitSystem::measure::pressure, value);
        } else if (spec.actual_name == "SWAT") {
            value = getValue(fs.saturation(waterPhaseIdx));
        } else if (spec.actual_name == "SGAS") {
            value = getValue(fs.saturation(gasPhaseIdx));
        } else if (spec.actual_name == "SOIL") {
            value = getValue(fs.saturation(oilPhaseIdx));
        } else if (spec.actual_name == "RS") {
            value = getValue(fs.Rs());
            value = unitSyst.from_si(UnitSystem::measure::gas_oil_ratio, value);
        } else if (spec.actual_name == "RV") {
            value = getValue(fs.Rv());
            value = unitSyst.from_si(UnitSystem::measure::oil_gas_ratio, value);
        } else if (spec.actual_name == "PERMX") {
            const auto& eclState = simulator_.vanguard().eclState();
            const auto& fp = eclState.fieldProps();
            auto permX = fp.get_double("PERMX");
            value = permX[cell_index];
            value = unitSyst.from_si(UnitSystem::measure::permeability, value);
        } else {
            OPM_THROW(std::runtime_error, "Unknown per-cell feature: " + spec.actual_name);
        }

        Scalar transformed = spec.transform.apply(value);
        Scalar scaled = spec.scaler.scale(transformed);
        return scaled;
    }


    /*!
    * \brief Run the Hybrid Newton model to produce output predictions.
    *
    * Uses the input tensor (prepared by constructInputTensor) together
    * with the model specified in \p config to compute the corresponding
    * output tensor. Scaling and transformations defined in the configuration
    * are applied automatically.
    *
    * \param input The input tensor of shape
    *              [(# of scalar features) + (# of per-cell features × n_cells)].
    * \param config The HybridNewtonConfig specifying model path and output
    *        feature definitions.
    * \return A tensor of shape [n_cells x n_output_features], where rows
    *         correspond to cells and columns correspond to output features.
    *
    * \throws std::runtime_error if model inference fails or if the
    *         output tensor does not match the expected feature layout.
    */
    ML::Tensor<Evaluation>
    constructOutputTensor(const ML::Tensor<Evaluation>& input,
                          const HybridNewtonConfig& config)
    {
        const auto& features = config.output_features;
        const std::size_t output_tensor_length = config.n_cells * features.size();

        ML::NNModel<Evaluation> model;
        const auto data_type = (config.model_data_type == "float64")
            ? ML::ModelDataType::kFloat64
            : ML::ModelDataType::kFloat32;
        model.loadModel(config.model_path, data_type);

        ML::Tensor<Evaluation> output(1, output_tensor_length);
        model.apply(input, output);

        return output;
    }

    /*!
    * \brief Update the nonlinear solver's initial guess using ML predictions.
    *
    * For each target cell, this function:
    *   1. Reads predicted output features from the ML tensor,
    *   2. Applies inverse scaling and transformations,
    *   3. If marked as delta values (`spec.is_delta == true`),
    *      adds the predicted delta to the current simulator state,
    *   4. Updates fluid state variables accordingly:
    *        - PRESSURE: replaces oil-phase pressure and reconstructs
    *          other phase pressures using capillary pressure relations,
    *        - SWAT, SOIL, SGAS: updates saturations with closure rules
    *          if fewer than 3 are predicted,
    *        - RS, RV: updates dissolved/volatilized gas-oil ratios
    *          if composition-switching is enabled.
    *
    * Special handling:
    *   - Saturations are clamped to [0,1] and normalized to ensure
    *     their sum is unity.
    *   - Invalid states (e.g., zero saturation sum) raise exceptions.
    *
    * \param output The ML output tensor [n_cells x n_output_features].
    * \param config The model configuration specifying output feature mappings.
    *
    * \throws std::runtime_error if an unknown output feature is encountered
    *         or if state consistency cannot be enforced.
    */
protected:
    Simulator& simulator_;
    BlackOilHybridNewtonReservoirInit<TypeTag> reservoirInit_;
    std::vector<HybridNewtonConfig> configs_;
    bool configsLoaded_;
};

} // namespace Opm

#endif // HYBRID_NEWTON_CLASS_HPP
