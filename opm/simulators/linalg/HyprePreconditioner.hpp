/*
  Copyright 2024 SINTEF AS
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

#ifndef OPM_HYPRE_PRECONDITIONER_HEADER_INCLUDED
#define OPM_HYPRE_PRECONDITIONER_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_krylov.h>
#include <_hypre_utilities.h>

#include <vector>
#include <numeric>

namespace Hypre {

/**
 * @brief Wrapper for Hypre's BoomerAMG preconditioner.
 *
 * This class provides an interface to the BoomerAMG preconditioner from the Hypre library.
 * It is designed to work with matrices, update vectors, and defect vectors specified by the template parameters.
 *
 * @tparam M The matrix type the preconditioner is for.
 * @tparam X The type of the update vector.
 * @tparam Y The type of the defect vector.
 */
template<class M, class X, class Y>
class HyprePreconditioner : public Dune::PreconditionerWithUpdate<X,Y> {
public:

    /**
     * @brief Constructor for the HyprePreconditioner class.
     *
     * Initializes the preconditioner with the given matrix and property tree.
     *
     * @param A The matrix for which the preconditioner is constructed.
     * @param prm The property tree containing configuration parameters.
     */
    HyprePreconditioner (const M& A, const Opm::PropertyTree prm)
        : A_(A)
    {
        OPM_TIMEBLOCK(prec_construct);

        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            OPM_THROW(std::runtime_error, "HyprePreconditioner is currently only implemented for sequential runs");
        }

        use_gpu_ = prm.get<bool>("use_gpu", false);

        // Set memory location and execution policy
#if HYPRE_USING_CUDA || HYPRE_USING_HIP
        if (use_gpu_) {
            HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
            HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
            // use hypre's SpGEMM instead of vendor implementation
            HYPRE_SetSpGemmUseVendor(false);
            // use cuRand for PMIS
            HYPRE_SetUseGpuRand(1);
            HYPRE_DeviceInitialize();
            HYPRE_PrintDeviceInfo();
        }
        else
#endif
        {
            HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
            HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
        }

        // Create the solver (BoomerAMG)
        HYPRE_BoomerAMGCreate(&solver_);

        // Set parameters from property tree with defaults
        HYPRE_BoomerAMGSetPrintLevel(solver_, prm.get<int>("print_level", 0));
        HYPRE_BoomerAMGSetMaxIter(solver_, prm.get<int>("max_iter", 1));
        HYPRE_BoomerAMGSetStrongThreshold(solver_, prm.get<double>("strong_threshold", 0.5));
        HYPRE_BoomerAMGSetAggTruncFactor(solver_, prm.get<double>("agg_trunc_factor", 0.3));
        HYPRE_BoomerAMGSetInterpType(solver_, prm.get<int>("interp_type", 6));
        HYPRE_BoomerAMGSetMaxLevels(solver_, prm.get<int>("max_levels", 15));
        HYPRE_BoomerAMGSetTol(solver_, prm.get<double>("tolerance", 0.0));

        if (use_gpu_) {
            HYPRE_BoomerAMGSetRelaxType(solver_, 16);
            HYPRE_BoomerAMGSetCoarsenType(solver_, 8);
            HYPRE_BoomerAMGSetAggNumLevels(solver_, 0);
            HYPRE_BoomerAMGSetAggInterpType(solver_, 6);
            // Keep transpose to avoid SpMTV
            HYPRE_BoomerAMGSetKeepTranspose(solver_, true);
        }
        else {
            HYPRE_BoomerAMGSetRelaxType(solver_, prm.get<int>("relax_type", 13));
            HYPRE_BoomerAMGSetCoarsenType(solver_, prm.get<int>("coarsen_type", 10));
            HYPRE_BoomerAMGSetAggNumLevels(solver_, prm.get<int>("agg_num_levels", 1));
            HYPRE_BoomerAMGSetAggInterpType(solver_, prm.get<int>("agg_interp_type", 4));
        }

        // Create Hypre vectors
        N_ = A_.N();
        nnz_ = A_.nonzeroes();
        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, N_-1, &x_hypre_);
        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, N_-1, &b_hypre_);
        HYPRE_IJVectorSetObjectType(x_hypre_, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(b_hypre_, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x_hypre_);
        HYPRE_IJVectorInitialize(b_hypre_);
        // Create indices vector
        indices_.resize(N_);
        std::iota(indices_.begin(), indices_.end(), 0);
        if (use_gpu_) {
            indices_device_ = hypre_CTAlloc(HYPRE_BigInt, N_, HYPRE_MEMORY_DEVICE);
            hypre_TMemcpy(indices_device_, indices_.data(), HYPRE_BigInt, N_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            // Allocate device vectors
            x_values_device_ = hypre_CTAlloc(HYPRE_Real, N_, HYPRE_MEMORY_DEVICE);
            b_values_device_ = hypre_CTAlloc(HYPRE_Real, N_, HYPRE_MEMORY_DEVICE);
        }

        // Create Hypre matrix
        HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, N_-1, 0, N_-1, &A_hypre_);
        HYPRE_IJMatrixSetObjectType(A_hypre_, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_hypre_);

        setupSparsityPattern();
        update();
    }

    /**
     * @brief Destructor for the HyprePreconditioner class.
     *
     * Cleans up resources allocated by the preconditioner.
     */
    ~HyprePreconditioner() {
        if (solver_) {
            HYPRE_BoomerAMGDestroy(solver_);
        }
        if (A_hypre_) {
            HYPRE_IJMatrixDestroy(A_hypre_);
        }
        if (x_hypre_) {
            HYPRE_IJVectorDestroy(x_hypre_);
        }
        if (b_hypre_) {
            HYPRE_IJVectorDestroy(b_hypre_);
        }
        if (values_device_) {
            hypre_TFree(values_device_, HYPRE_MEMORY_DEVICE);
        }
        if (x_values_device_) {
            hypre_TFree(x_values_device_, HYPRE_MEMORY_DEVICE);
        }
        if (b_values_device_) {
            hypre_TFree(b_values_device_, HYPRE_MEMORY_DEVICE);
        }
        if (indices_device_) {
            hypre_TFree(indices_device_, HYPRE_MEMORY_DEVICE);
        }
    }

    /**
     * @brief Updates the preconditioner with the current matrix values.
     *
     * This method should be called whenever the matrix values change.
     */
    void update() override {
        OPM_TIMEBLOCK(prec_update);
        copyMatrixToHypre();
        HYPRE_BoomerAMGSetup(solver_, parcsr_A_, par_b_, par_x_);
    }

    /**
     * @brief Pre-processing step before applying the preconditioner.
     *
     * This method is currently a no-op.
     *
     * @param v The update vector.
     * @param d The defect vector.
     */
    void pre(X& /*v*/, Y& /*d*/) override {
    }

    /**
     * @brief Applies the preconditioner to a vector.
     *
     * Performs one AMG V-cycle to solve the system.
     *
     * @param v The update vector.
     * @param d The defect vector.
     */
    void apply(X& v, const Y& d) override {
        OPM_TIMEBLOCK(prec_apply);

        // Copy vectors to Hypre format
        copyVectorsToHypre(v, d);

        // Apply the preconditioner (one AMG V-cycle)
        HYPRE_BoomerAMGSolve(solver_, parcsr_A_, par_b_, par_x_);

        // Copy result back
        copyVectorFromHypre(v);
    }

    /**
     * @brief Post-processing step after applying the preconditioner.
     *
     * This method is currently a no-op.
     *
     * @param v The update vector.
     */
    void post(X& /*v*/) override {
    }

    /**
     * @brief Returns the solver category.
     *
     * @return The solver category, which is sequential.
     */
    Dune::SolverCategory::Category category() const override {
        return Dune::SolverCategory::sequential;
    }

    /**
     * @brief Checks if the preconditioner has a perfect update.
     *
     * @return True, indicating that the preconditioner can be perfectly updated.
     */
    bool hasPerfectUpdate() const override
    {
        // The Hypre preconditioner can depend on the values of the matrix so it does not have perfect update.
        // However, copying the matrix to Hypre requires to setup the solver again, so this is handled internally.
        // So for ISTLSolver, we can return true.
        return true;
    }

private:
    /**
     * @brief Sets up the sparsity pattern for the Hypre matrix.
     *
     * Allocates and initializes arrays required by Hypre.
     */
    void setupSparsityPattern() {
        // Allocate arrays required by Hypre
        ncols_.resize(N_);
        rows_.resize(N_);
        cols_.resize(nnz_);

        // Setup arrays and fill column indices
        int pos = 0;
        for (auto row = A_.begin(); row != A_.end(); ++row) {
            const int rowIdx = row.index();
            rows_[rowIdx] = rowIdx;
            ncols_[rowIdx] = row->size();

            for (auto col = row->begin(); col != row->end(); ++col) {
                cols_[pos++] = col.index();
            }
        }
        if (use_gpu_) {
            // Allocate device arrays
            ncols_device_ = hypre_CTAlloc(HYPRE_Int, N_, HYPRE_MEMORY_DEVICE);
            rows_device_ = hypre_CTAlloc(HYPRE_BigInt, N_, HYPRE_MEMORY_DEVICE);
            cols_device_ = hypre_CTAlloc(HYPRE_BigInt, nnz_, HYPRE_MEMORY_DEVICE);
            values_device_ = hypre_CTAlloc(HYPRE_Real, nnz_, HYPRE_MEMORY_DEVICE);

            // Copy to device
            hypre_TMemcpy(ncols_device_, ncols_.data(), HYPRE_Int, N_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(rows_device_, rows_.data(), HYPRE_BigInt, N_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(cols_device_, cols_.data(), HYPRE_BigInt, nnz_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
        }
    }

    /**
     * @brief Copies the matrix values to the Hypre matrix.
     *
     * This method transfers the matrix data from the host to the Hypre matrix.
     * It assumes that the values of the matrix are stored in a contiguous array.
     * If GPU is used, the data is transferred to the device.
     */
    void copyMatrixToHypre() {
        OPM_TIMEBLOCK(prec_copy_matrix);
        // Get pointer to matrix values array
        const HYPRE_Real* values = &(A_[0][0][0][0]);
        // Indexing explanation:
        // A_[0]             - First row of the matrix
        //     [0]           - First block in that row
        //        [0]        - First row within the 1x1 block
        //           [0]     - First column within the 1x1 block

        if (use_gpu_) {
            hypre_TMemcpy(values_device_, values, HYPRE_Real, nnz_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            HYPRE_IJMatrixSetValues(A_hypre_, N_, ncols_device_, rows_device_, cols_device_, values_device_);
        }
        else {
            HYPRE_IJMatrixSetValues(A_hypre_, N_, ncols_.data(), rows_.data(), cols_.data(), values);
        }

        HYPRE_IJMatrixAssemble(A_hypre_);
        HYPRE_IJMatrixGetObject(A_hypre_, reinterpret_cast<void**>(&parcsr_A_));
    }

    /**
     * @brief Copies vectors to the Hypre format.
     *
     * Transfers the update and defect vectors to Hypre.
     * If GPU is used, the data is transferred from the host to the device.
     *
     * @param v The update vector.
     * @param d The defect vector.
     */
    void copyVectorsToHypre(const X& v, const Y& d) {
        OPM_TIMEBLOCK(prec_copy_vectors_to_hypre);
        const HYPRE_Real* x_vals = &(v[0][0]);
        const HYPRE_Real* b_vals = &(d[0][0]);

        if (use_gpu_) {
            hypre_TMemcpy(x_values_device_, x_vals, HYPRE_Real, N_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(b_values_device_, b_vals, HYPRE_Real, N_, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

            HYPRE_IJVectorSetValues(x_hypre_, N_, indices_device_, x_values_device_);
            HYPRE_IJVectorSetValues(b_hypre_, N_, indices_device_, b_values_device_);
        }
        else {
            HYPRE_IJVectorSetValues(x_hypre_, N_, indices_.data(), x_vals);
            HYPRE_IJVectorSetValues(b_hypre_, N_, indices_.data(), b_vals);
        }

        HYPRE_IJVectorAssemble(x_hypre_);
        HYPRE_IJVectorAssemble(b_hypre_);
        HYPRE_IJVectorGetObject(x_hypre_, reinterpret_cast<void**>(&par_x_));
        HYPRE_IJVectorGetObject(b_hypre_, reinterpret_cast<void**>(&par_b_));
    }

    /**
     * @brief Copies the solution vector from Hypre.
     *
     * Transfers the solution vector from Hypre back to the host.
     * If GPU is used, the data is transferred from the device to the host.
     *
     * @param v The update vector.
     */
    void copyVectorFromHypre(X& v) {
        OPM_TIMEBLOCK(prec_copy_vector_from_hypre);
        HYPRE_Real* values = &(v[0][0]);
        if (use_gpu_) {
            HYPRE_IJVectorGetValues(x_hypre_, N_, indices_device_, x_values_device_);
            hypre_TMemcpy(values, x_values_device_, HYPRE_Real, N_, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
        }
        else {
            HYPRE_IJVectorGetValues(x_hypre_, N_, indices_.data(), values);
        }
    }

    const M& A_; //!< The matrix for which the preconditioner is constructed.
    bool use_gpu_ = false; //!< Flag indicating whether to use GPU acceleration.

    HYPRE_Solver solver_ = nullptr; //!< The Hypre solver object.
    HYPRE_IJMatrix A_hypre_ = nullptr; //!< The Hypre matrix object.
    HYPRE_ParCSRMatrix parcsr_A_ = nullptr; //!< The parallel CSR matrix object.
    HYPRE_IJVector x_hypre_ = nullptr; //!< The Hypre solution vector.
    HYPRE_IJVector b_hypre_ = nullptr; //!< The Hypre right-hand side vector.
    HYPRE_ParVector par_x_ = nullptr; //!< The parallel solution vector.
    HYPRE_ParVector par_b_ = nullptr; //!< The parallel right-hand side vector.

    std::vector<HYPRE_Int> ncols_; //!< Number of columns per row.
    std::vector<HYPRE_BigInt> rows_; //!< Row indices.
    std::vector<HYPRE_BigInt> cols_; //!< Column indices.
    HYPRE_Int* ncols_device_ = nullptr; //!< Device array for number of columns per row.
    HYPRE_BigInt* rows_device_ = nullptr; //!< Device array for row indices.
    HYPRE_BigInt* cols_device_ = nullptr; //!< Device array for column indices.
    HYPRE_Real* values_device_ = nullptr; //!< Device array for matrix values.

    std::vector<HYPRE_BigInt> indices_; //!< Indices vector for copying vectors to/from Hypre.
    HYPRE_BigInt* indices_device_ = nullptr; //!< Device array for indices.
    HYPRE_Int N_ = -1; //!< Number of rows in the matrix.
    HYPRE_Int nnz_ = -1; //!< Number of non-zero elements in the matrix.

    HYPRE_Real* x_values_device_ = nullptr; //!< Device array for solution vector values.
    HYPRE_Real* b_values_device_ = nullptr; //!< Device array for right-hand side vector values.
};

} // namespace Hypre

#endif
