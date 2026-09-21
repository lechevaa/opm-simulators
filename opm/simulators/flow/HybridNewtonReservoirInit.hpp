// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2026 NORCE Research AS

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

#ifndef HYBRID_NEWTON_RESERVOIR_INIT_HPP
#define HYBRID_NEWTON_RESERVOIR_INIT_HPP

#include <opm/simulators/flow/FlowBaseProblemProperties.hpp>
#include <opm/simulators/flow/HybridNewtonConfig.hpp>
#include <opm/input/eclipse/Units/UnitSystem.hpp>
#include <opm/models/blackoil/blackoilnewtonmethodparams.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/ml/ml_model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace Opm {

template <typename TypeTag>
class BlackOilHybridNewtonReservoirInit
{
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using MaterialLaw = GetPropType<TypeTag, Properties::MaterialLaw>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;

    enum { numPhases = FluidSystem::numPhases };
    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };
    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };
    static constexpr bool compositionSwitchEnabled = Indices::compositionSwitchIdx >= 0;

    struct FeatureFlags
    {
        bool has_SWAT = false;
        bool has_SOIL = false;
        bool has_SGAS = false;
        bool has_PRESSURE = false;
    };

public:
    explicit BlackOilHybridNewtonReservoirInit(Simulator& simulator)
        : simulator_(simulator)
    {}

    template <class ConstructInputFn,
              class ConstructOutputFn,
              class WriteProvenanceFn>
    void runReservoirHybridNewton(const HybridNewtonConfig& config,
                                  ConstructInputFn&& construct_input,
                                  ConstructOutputFn&& construct_output,
                                  WriteProvenanceFn&& write_provenance)
    {
        auto input = construct_input(config);
        auto output = construct_output(input, config);
        write_provenance(config, input, output);
        updateInitialGuess(output, config);
    }

private:
    void updateInitialGuess(ML::Tensor<Evaluation>& output,
                            const HybridNewtonConfig& config)
    {
        const auto& features = config.output_features;
        BlackoilNewtonParams<Scalar> bparams;
        bparams.read();

        FeatureFlags flags = flagFeatures(features);
        const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
        std::vector<Scalar> target_sw(config.n_cells, nan);
        std::vector<Scalar> target_so(config.n_cells, nan);
        std::vector<Scalar> target_sg(config.n_cells, nan);
        std::vector<Scalar> target_po(config.n_cells, nan);
        std::vector<Scalar> target_rs(config.n_cells, nan);
        std::vector<Scalar> target_rv(config.n_cells, nan);

        const auto& unitSyst = simulator_.vanguard().schedule().getUnits();
        Scalar max_abs_pressure_update = 0.0;
        Scalar max_abs_saturation_update = 0.0;
        Scalar max_abs_rs_update = 0.0;
        Scalar max_abs_rv_update = 0.0;

        for (std::size_t i = 0; i < config.n_cells; ++i) {
            const int cell_idx = config.cell_indices[i];
            const auto& intQuants = simulator_.model().intensiveQuantities(cell_idx, 0);
            auto fs = intQuants.fluidState();
            const auto old_sw = getValue(fs.saturation(waterPhaseIdx));
            const auto old_so = getValue(fs.saturation(oilPhaseIdx));
            const auto old_sg = getValue(fs.saturation(gasPhaseIdx));
            const auto old_po = getValue(fs.pressure(oilPhaseIdx));
            const auto old_rs = getValue(fs.Rs());
            const auto old_rv = getValue(fs.Rv());

            int feature_idx = 0;
            Scalar sw_val = -1.0;
            Scalar so_val = -1.0;
            Scalar sg_val = -1.0;
            Scalar po_val = -1.0;

            for (const auto& [name, spec] : features) {
                auto scaled_value = getValue(output(feature_idx * config.n_cells + i));
                Scalar raw_value = spec.scaler.unscale(scaled_value);
                raw_value = spec.transform.applyInverse(raw_value);

                if (spec.is_delta) {
                    if (spec.actual_name == "PRESSURE") {
                        raw_value += unitSyst.from_si(UnitSystem::measure::pressure, getValue(fs.pressure(oilPhaseIdx)));
                    } else if (spec.actual_name == "SWAT") {
                        raw_value += getValue(fs.saturation(waterPhaseIdx));
                    } else if (spec.actual_name == "SOIL") {
                        raw_value += getValue(fs.saturation(oilPhaseIdx));
                    } else if (spec.actual_name == "SGAS") {
                        raw_value += getValue(fs.saturation(gasPhaseIdx));
                    } else if (spec.actual_name == "RS") {
                        raw_value += unitSyst.from_si(UnitSystem::measure::gas_oil_ratio, getValue(fs.Rs()));
                    } else if (spec.actual_name == "RV") {
                        raw_value += unitSyst.from_si(UnitSystem::measure::oil_gas_ratio, getValue(fs.Rv()));
                    } else {
                        OPM_THROW(std::runtime_error, "Unknown delta feature: " + name);
                    }
                }

                if (spec.actual_name == "PRESSURE") {
                    po_val = config.output_values_are_si
                        ? raw_value
                        : unitSyst.to_si(UnitSystem::measure::pressure, raw_value);
                } else if (spec.actual_name == "SWAT") {
                    sw_val = raw_value;
                } else if (spec.actual_name == "SOIL") {
                    so_val = raw_value;
                } else if (spec.actual_name == "SGAS") {
                    sg_val = raw_value;
                } else if (spec.actual_name == "RS") {
                    if constexpr (compositionSwitchEnabled) {
                        if (!config.output_values_are_si) {
                            raw_value = unitSyst.to_si(UnitSystem::measure::gas_oil_ratio, raw_value);
                        }
                        max_abs_rs_update = std::max(max_abs_rs_update, std::abs(raw_value - old_rs));
                        fs.setRs(raw_value);
                    }
                } else if (spec.actual_name == "RV") {
                    if constexpr (compositionSwitchEnabled) {
                        if (!config.output_values_are_si) {
                            raw_value = unitSyst.to_si(UnitSystem::measure::oil_gas_ratio, raw_value);
                        }
                        max_abs_rv_update = std::max(max_abs_rv_update, std::abs(raw_value - old_rv));
                        fs.setRv(raw_value);
                    }
                } else {
                    OPM_THROW(std::runtime_error, "Unknown output feature: " + name);
                }

                ++feature_idx;
            }

            const int sat_count = static_cast<int>(flags.has_SWAT) +
                                  static_cast<int>(flags.has_SOIL) +
                                  static_cast<int>(flags.has_SGAS);

            if (sat_count >= 2) {
                Scalar sw = sw_val;
                Scalar so = so_val;
                Scalar sg = sg_val;

                if (!flags.has_SWAT) {
                    sw = 1.0 - so - sg;
                } else if (!flags.has_SOIL) {
                    so = 1.0 - sw - sg;
                } else if (!flags.has_SGAS) {
                    sg = 1.0 - sw - so;
                }

                sw = snapSaturationNearBounds(sw);
                so = snapSaturationNearBounds(so);
                sg = snapSaturationNearBounds(sg);

                sw = std::max(0.0, std::min(sw, 1.0));
                so = std::max(0.0, std::min(so, 1.0));
                sg = std::max(0.0, std::min(sg, 1.0));

                const Scalar sum = sw + so + sg;
                if (sum <= 1e-12) {
                    OPM_THROW(std::runtime_error, "Saturation sum is zero in cell " + std::to_string(cell_idx));
                }

                const auto sw_final = sw / sum;
                const auto so_final = so / sum;
                const auto sg_final = sg / sum;
                fs.setSaturation(waterPhaseIdx, sw_final);
                fs.setSaturation(oilPhaseIdx, so_final);
                fs.setSaturation(gasPhaseIdx, sg_final);
                target_sw[i] = sw_final;
                target_so[i] = so_final;
                target_sg[i] = sg_final;
            }

            if (flags.has_PRESSURE) {
                std::array<Evaluation, numPhases> pC;
                const auto& materialParams = simulator_.problem().materialLawParams(cell_idx);
                MaterialLaw::capillaryPressures(pC, materialParams, fs);

                for (unsigned phaseIdx = 0; phaseIdx < numPhases; ++phaseIdx) {
                    if (!FluidSystem::phaseIsActive(phaseIdx)) {
                        continue;
                    }
                    if (phaseIdx == oilPhaseIdx) {
                        fs.setPressure(phaseIdx, po_val);
                    } else {
                        fs.setPressure(phaseIdx, po_val + (getValue(pC[phaseIdx]) - getValue(pC[oilPhaseIdx])));
                    }
                }
                target_po[i] = po_val;
            }

            if constexpr (compositionSwitchEnabled) {
                if (std::ranges::any_of(features, [](const auto& feature) { return feature.second.actual_name == "RS"; })) {
                    target_rs[i] = getValue(fs.Rs());
                }
                if (std::ranges::any_of(features, [](const auto& feature) { return feature.second.actual_name == "RV"; })) {
                    target_rv[i] = getValue(fs.Rv());
                }
            }

            max_abs_saturation_update = std::max(max_abs_saturation_update, std::abs(getValue(fs.saturation(waterPhaseIdx)) - old_sw));
            max_abs_saturation_update = std::max(max_abs_saturation_update, std::abs(getValue(fs.saturation(oilPhaseIdx)) - old_so));
            max_abs_saturation_update = std::max(max_abs_saturation_update, std::abs(getValue(fs.saturation(gasPhaseIdx)) - old_sg));
            max_abs_pressure_update = std::max(max_abs_pressure_update, std::abs(getValue(fs.pressure(oilPhaseIdx)) - old_po));

            auto& primaryVars = simulator_.model().solution(0)[cell_idx];
            primaryVars.assignNaive(fs);
        }

        simulator_.model().invalidateAndUpdateIntensiveQuantities(0);

        std::ostringstream os;
        os.setf(std::ios::scientific);
        os.precision(6);
        os << "HybridNewton model=" << config.model_path
           << " dtype=" << config.model_data_type
           << " max|dP_oil(SI)|=" << max_abs_pressure_update
           << " max|dS|=" << max_abs_saturation_update
           << " max|dRs(SI)|=" << max_abs_rs_update
           << " max|dRv(SI)|=" << max_abs_rv_update;
        Opm::OpmLog::debug(os.str());
    }

    static constexpr Scalar hybridSaturationRoundoffTolerance()
    {
        return Scalar{1e-16};
    }

    static Scalar snapSaturationNearBounds(Scalar value)
    {
        const Scalar tol = hybridSaturationRoundoffTolerance();
        if (std::abs(value) <= tol) {
            return Scalar{0.0};
        }
        if (std::abs(value - Scalar{1.0}) <= tol) {
            return Scalar{1.0};
        }
        return value;
    }

    static FeatureFlags flagFeatures(const std::vector<std::pair<std::string, FeatureSpec>>& features)
    {
        FeatureFlags flags;
        for (const auto& [name, spec] : features) {
            if (spec.actual_name == "SWAT") {
                flags.has_SWAT = true;
            } else if (spec.actual_name == "SOIL") {
                flags.has_SOIL = true;
            } else if (spec.actual_name == "SGAS") {
                flags.has_SGAS = true;
            } else if (spec.actual_name == "PRESSURE") {
                flags.has_PRESSURE = true;
            }
        }
        return flags;
    }

    Simulator& simulator_;
};

} // namespace Opm

#endif // HYBRID_NEWTON_RESERVOIR_INIT_HPP
