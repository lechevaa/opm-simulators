// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief This file serves as an example of how to use the tasklet mechanism for
 *        asynchronous work.
 */
#include "config.h"

#include <ewoms/parallel/tasklets.hh>

#include <chrono>
#include <iostream>

class SleepTasklet : public Ewoms::TaskletInterface
{
public:
    SleepTasklet(int mseconds)
        : mseconds_(mseconds)
    {
        n_ = numInstantiated_;
        ++ numInstantiated_;
    }

    void run()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mseconds_));
        std::cout << "Sleep tasklet " << n_ << " of " << mseconds_ << " ms finished" << std::endl;
    }

private:
    static int numInstantiated_;
    int n_;
    int mseconds_;
};

int SleepTasklet::numInstantiated_ = 0;

int main()
{
    Ewoms::TaskletRunner tr(2);

    for (int i = 0; i < 5; ++ i) {
        //auto st = std::make_shared<SleepTasklet>((i + 1)*1000);
        auto st = std::make_shared<SleepTasklet>(100);
        tr.dispatch(st);
    }

    std::cout << "before barrier" << std::endl;
    tr.barrier();
    std::cout << "after barrier" << std::endl;

    for (int i = 0; i < 7; ++ i) {
        auto st = std::make_shared<SleepTasklet>(500);
        tr.dispatch(st);
    }

    return 0;
}

