/// ILU apply part 2: backward substitution.
/// Solves U*x=y where U is an upper triangular sparse blocked matrix.
/// Here, U is it's own BSR matrix.
/// Only used with ChowPatelIlu.
__kernel void ILU_apply2(
    __global const double *LUvals,
    __global const int *LUcols,
    __global const int *LUrows,
    __global const int *diagIndex,
    __global const double *invDiagVals,
    __global double *x,
    __global const unsigned int *nodesPerColorPrefix,
    const unsigned int color,
    const unsigned int block_size,
    __local double *tmp)
{
    const unsigned int warpsize = 32;
    const unsigned int bs = block_size;
    const unsigned int idx_t = get_local_id(0);
    const unsigned int num_active_threads = (warpsize/bs/bs)*bs*bs;
    const unsigned int num_blocks_per_warp = warpsize/bs/bs;
    const unsigned int NUM_THREADS = get_global_size(0);
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    unsigned int idx_g = get_global_id(0);
    unsigned int target_block_row = idx_g / warpsize;
    target_block_row += nodesPerColorPrefix[color];
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bs) % bs;
    const unsigned int r = lane % bs;

    while(target_block_row < nodesPerColorPrefix[color+1]){
        const unsigned int first_block = LUrows[target_block_row];
        const unsigned int last_block = LUrows[target_block_row+1];
        unsigned int block = first_block + lane / (bs*bs);
        double local_out = 0.0;

        if(lane < num_active_threads){
            if(lane < bs){
                const unsigned int row = target_block_row*bs+lane;
                local_out = x[row];
            }
            for(; block < last_block; block += num_blocks_per_warp){
                const double x_elem = x[LUcols[block]*bs + c];
                const double A_elem = LUvals[block*bs*bs + c + r*bs];
                local_out -= x_elem * A_elem;
            }
        }

        // do reduction in shared mem
        tmp[lane] = local_out;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(unsigned int offset = 3; offset <= 24; offset <<= 1)
        {
            if (lane + offset < warpsize)
            {
                tmp[lane] += tmp[lane + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_out = tmp[lane];

        if(lane < bs){
            tmp[lane + bs*idx_t/warpsize] = local_out;
            double sum = 0.0;
            for(int i = 0; i < bs; ++i){
                sum += invDiagVals[target_block_row*bs*bs + i + lane*bs] * tmp[i + bs*idx_t/warpsize];
            }

            const unsigned int row = target_block_row*bs + lane;
            x[row] = sum;
        }

        target_block_row += num_warps_in_grid;
    }
}
