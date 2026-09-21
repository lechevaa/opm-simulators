/*
  Copyright 2026 NORCE

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

#include <config.h>

#define BOOST_TEST_MODULE TestHybridNewtonConfig

#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/HybridNewtonConfig.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/common/utility/FileSystem.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

class TempFile
{
public:
    TempFile()
        : fname_ { std::filesystem::temp_directory_path() /
                   Opm::unique_path("hybrid-newton-cells-%%%%") }
    {}

    ~TempFile()
    {
        std::filesystem::remove(fname_);
    }

    const std::filesystem::path& name() const
    {
        return fname_;
    }

private:
    std::filesystem::path fname_;
};

Opm::PropertyTree makeConfigTree(const std::string& cell_file,
                                 const std::string& data_type = "")
{
    Opm::PropertyTree pt;
    pt.put("model_path", std::string{"models/test.model"});
    if (!data_type.empty()) {
        pt.put("model_data_type", data_type);
    }
    pt.put("cell_indices_file", cell_file);
    pt.put("apply_times.0", 0.0);
    return pt;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(Default_ModelDataType_Is_Float32)
{
    TempFile cells;
    {
        std::ofstream out(cells.name());
        out << "0\n1\n";
    }

    auto pt = makeConfigTree(cells.name().string());
    Opm::HybridNewtonConfig config(pt);
    BOOST_CHECK_EQUAL(config.model_data_type, "float32");
}

BOOST_AUTO_TEST_CASE(Explicit_ModelDataType_Float64_Is_Accepted)
{
    TempFile cells;
    {
        std::ofstream out(cells.name());
        out << "0\n";
    }

    auto pt = makeConfigTree(cells.name().string(), "float64");
    Opm::HybridNewtonConfig config(pt);
    BOOST_CHECK_EQUAL(config.model_data_type, "float64");
}

BOOST_AUTO_TEST_CASE(Invalid_ModelDataType_Throws)
{
    TempFile cells;
    {
        std::ofstream out(cells.name());
        out << "0\n";
    }

    auto pt = makeConfigTree(cells.name().string(), "float16");
    BOOST_CHECK_THROW(Opm::HybridNewtonConfig config(pt), std::runtime_error);
}
