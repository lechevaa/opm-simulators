/*
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
#include "config.h"

#include <flow/flow_ebos_blackoil_tpfa.hpp>

#include <opm/material/common/ResetLocale.hpp>
#include <opm/grid/CpGrid.hpp>
#include <opm/simulators/flow/SimulatorFullyImplicitBlackoilEbos.hpp>
#include <opm/simulators/flow/Main.hpp>

#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

namespace Opm {
    namespace Properties {
        namespace TTag {
            struct EclFlowProblemTPFA {
            using InheritsFrom = std::tuple<EclFlowProblem>;
            };
        }
   }
}

namespace Opm {
    namespace Properties {

        template<class TypeTag>
        struct Linearizer<TypeTag, TTag::EclFlowProblemTPFA> { using type = TpfaLinearizer<TypeTag>; };

        template<class TypeTag>
        struct LocalResidual<TypeTag, TTag::EclFlowProblemTPFA> { using type = BlackOilLocalResidualTPFA<TypeTag>; };

        template<class TypeTag>
        struct EnableDiffusion<TypeTag, TTag::EclFlowProblemTPFA> { static constexpr bool value = false; };

    }
}


namespace Opm
{

// ----------------- Main program -----------------
int flowEbosBlackoilTpfaMain(int argc, char** argv, bool outputCout, bool outputFiles)
{
    // we always want to use the default locale, and thus spare us the trouble
    // with incorrect locale settings.
    resetLocale();

    FlowMainEbos<Properties::TTag::EclFlowProblemTPFA>
        mainfunc {argc, argv, outputCout, outputFiles};
    return mainfunc.execute();
}

int flowEbosBlackoilTpfaMainStandalone(int argc, char** argv)
{
    using TypeTag = Properties::TTag::EclFlowProblemTPFA;
    auto mainObject = Opm::Main(argc, argv);
    return mainObject.runStatic<TypeTag>();
}

} // namespace Opm