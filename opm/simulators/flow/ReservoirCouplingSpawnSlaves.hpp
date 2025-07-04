/*
  Copyright 2024 Equinor ASA

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

#ifndef OPM_RESERVOIR_COUPLING_SPAWN_SLAVES_HPP
#define OPM_RESERVOIR_COUPLING_SPAWN_SLAVES_HPP

#include <opm/simulators/utils/ParallelCommunication.hpp>
#include <opm/simulators/flow/ReservoirCoupling.hpp>
#include <opm/simulators/flow/ReservoirCouplingMaster.hpp>

#include <opm/input/eclipse/Schedule/ResCoup/ReservoirCouplingInfo.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <mpi.h>

#include <filesystem>
#include <vector>

namespace Opm {

class ReservoirCouplingSpawnSlaves {
public:
    using MessageTag = ReservoirCoupling::MessageTag;

    ReservoirCouplingSpawnSlaves(
        ReservoirCouplingMaster &master,
        const ReservoirCoupling::CouplingInfo &rescoup
    );

    void spawn();

private:
    void createMasterGroupNameOrder_();
    void createMasterGroupToSlaveNameMap_();
    std::pair<std::vector<char>, std::size_t>
        getMasterGroupNamesForSlave_(const std::string &slave_name) const;
    std::vector<char *> getSlaveArgv_(
        const std::filesystem::path &data_file,
        const std::string &slave_name,
        std::string &log_filename) const;
    void prepareTimeStepping_();
    void receiveActivationDateFromSlaves_();
    void receiveSimulationStartDateFromSlaves_();
    void sendMasterGroupNamesToSlaves_();
    void sendSlaveNamesToSlaves_();
    void spawnSlaveProcesses_();

    ReservoirCouplingMaster &master_;
    const ReservoirCoupling::CouplingInfo &rescoup_;
    const Parallel::Communication &comm_;
    ReservoirCoupling::Logger logger_;
};

} // namespace Opm
#endif // OPM_RESERVOIR_COUPLING_SPAWN_SLAVES_HPP
