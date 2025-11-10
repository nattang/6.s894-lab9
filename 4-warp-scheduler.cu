// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 4: Bring Your Own Warp Scheduler
////////////////////////////////////////////////////////////////////////////////

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 4
#define ROUNDS_PER_BLOCK 4

#define ARR_M (132 * 10 * 8)
#define ARR_N (128 * 128 * 4)

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int N) {
    extern __shared__ __align__(128) unsigned char smem_raw[];              
    bf16 *tile_smem_a = reinterpret_cast<bf16*>(smem_raw);
    bf16 *tile_smem_b = tile_smem_a + (THREADS_PER_WARP * TILE_M * TILE_N);

    size_t tile_bytes = 2 * size_t(THREADS_PER_WARP) * TILE_M * TILE_N * sizeof(bf16);
    size_t mbar_off   = (tile_bytes + 8) & ~size_t(8);
    uint64_t *pmbar0 = reinterpret_cast<uint64_t*>(smem_raw + mbar_off);
    uint64_t *pmbar1 = pmbar0 + 1;
    uint64_t *cmbar0 = pmbar1 + 1; 
    uint64_t *cmbar1 = cmbar0 + 1; 

    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    int tile_id_y = blockIdx.y * TILE_M;

    int shared_offset = threadIdx.x * TILE_M * TILE_N;

    if (warp_id == 0 && lane == 0) {
        init_barrier(pmbar0, 32);
        init_barrier(pmbar1, 32);
        init_barrier(cmbar0, 32);
        init_barrier(cmbar1, 32);
        async_proxy_fence();
    }

    __syncthreads();

    for (uint round = 0; round < ROUNDS_PER_BLOCK; round++) {
        int tile_id_x = ((blockIdx.x * ROUNDS_PER_BLOCK * 32) + (round * 32) + threadIdx.x) * TILE_N;
        uint64_t *cur_pmbar = round & 1 ? pmbar1 : pmbar0;
        uint64_t *cur_cmbar = round & 1 ? cmbar1 : cmbar0;

        if (warp_id == 0) {
            if (round > 1) {
                wait(cur_pmbar, ((round / 2) + 1) % 2); // wait on the last producer operation to finish writing
                wait(cur_cmbar, ((round / 2) + 1) % 2); // wait on the previous round consumer operation to finish reading
            }

            expect_bytes_and_arrive(cur_pmbar, TILE_M*TILE_N*sizeof(bf16));

            __syncwarp();

            bf16 * base_tile_addr = round & 1 ? tile_smem_b : tile_smem_a;
            
            cp_async_bulk_tensor_2d_global_to_shared(&base_tile_addr[shared_offset], &tensor_map, tile_id_x, tile_id_y, cur_pmbar);

        } else if (warp_id == 1) {
            uint64_t *cur_pmbar = round & 1 ? pmbar1 : pmbar0;
            uint64_t *cur_cmbar = round & 1 ? cmbar1 : cmbar0;
            
            if (round > 1) {
                wait(cur_cmbar, ((round / 2) + 1) % 2);
            }

            wait(cur_pmbar, (round / 2) % 2);

            bf16 * base_tile_addr = round & 1 ? tile_smem_b : tile_smem_a;

            cp_async_bulk_tensor_2d_shared_to_global(&dest_tensor_map, tile_id_x, tile_id_y, &base_tile_addr[shared_offset]);

            if (lane == 0) {
                tma_commit_group();
            }
            
            __syncwarp();

            tma_wait_until_pending<0>();

            arrive(cur_cmbar, 1);

            __syncwarp();
        }

    }
}

void launch_multiwarp_pipeline(bf16 *dest, bf16 *src, const int N) {
    /*
     * IMPORTANT REQUIREMENT FOR PART 4:
     *
     * To receive credit for this part, you MUST launch the kernel with maximum
     * shared memory allocated.
     *
     * Use cudaFuncSetAttribute() with
     * cudaFuncAttributeMaxDynamicSharedMemorySize to configure the maximum
     * available shared memory before launching the kernel, and then **launch**
     * it with the maximum amount.
     */

    CUtensorMap src_map;

    const cuuint64_t global_dim[2] = {ARR_N, ARR_M};
    const cuuint64_t global_strides[1] = {ARR_N * sizeof(bf16)};
    const cuuint32_t box_dim[2] = {TILE_N,TILE_M};
    const cuuint32_t element_strides[2] = {1,1};

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &src_map, 
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, 
            src, 
            global_dim, 
            global_strides,
            box_dim, 
            element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );


    CUtensorMap dest_map; 

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &dest_map, 
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, 
            dest, 
            global_dim, 
            global_strides,
            box_dim, 
            element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );

    dim3 block_size(32,4);

    int num_blocks_m = CEIL_DIV(ARR_M, TILE_M);
    int num_blocks_n = CEIL_DIV(ARR_N, TILE_N * 32 * ROUNDS_PER_BLOCK);

    dim3 grid(num_blocks_n, num_blocks_m); // NOTE: n = x, m = y

    uint32_t shmem_size_bytes = 227000; // ((THREADS_PER_WARP * TILE_M * TILE_N * sizeof(bf16)) + sizeof(uint64_t));

    CUDA_CHECK(cudaFuncSetAttribute(
            tma_multiwarp_pipeline,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_bytes));

    tma_multiwarp_pipeline<<<grid, block_size, shmem_size_bytes>>>(src_map, dest_map, N);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

const int elem_per_block = 16384;
__global__ void simple_vector_copy(bf16 *__restrict__ dest,
                                   const bf16 *__restrict__ src, int N) {
    constexpr int VEC_ELEMS = 8;
    using VecT = uint4;

    int total_vecs = elem_per_block / VEC_ELEMS;
    int start_vec = (blockIdx.x * blockDim.x) * total_vecs;

    const VecT *src_vec = reinterpret_cast<const VecT *>(src);
    VecT *dest_vec = reinterpret_cast<VecT *>(dest);

    for (int i = threadIdx.x; i < blockDim.x * total_vecs; i += blockDim.x) {
        dest_vec[start_vec + i] = src_vec[start_vec + i];
    }
}

#define BENCHMARK_KERNEL(kernel_call, num_iters, size_bytes, label)            \
    do {                                                                       \
        cudaEvent_t start, stop;                                               \
        CUDA_CHECK(cudaEventCreate(&start));                                   \
        CUDA_CHECK(cudaEventCreate(&stop));                                    \
        CUDA_CHECK(cudaEventRecord(start));                                    \
        for (int i = 0; i < num_iters; i++) {                                  \
            kernel_call;                                                       \
        }                                                                      \
        CUDA_CHECK(cudaEventRecord(stop));                                     \
        CUDA_CHECK(cudaEventSynchronize(stop));                                \
        float elapsed_time;                                                    \
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));          \
        float time_per_iter = elapsed_time / num_iters;                        \
        float bandwidth_gb_s = (2.0 * size_bytes * 1e-6 / time_per_iter);      \
        printf("%s - Time: %.4f ms, Bandwidth: %.2f GB/s\n", label,            \
               time_per_iter, bandwidth_gb_s);                                 \
        CUDA_CHECK(cudaEventDestroy(start));                                   \
        CUDA_CHECK(cudaEventDestroy(stop));                                    \
    } while (0)

int main() {
    const size_t size = 132 * 10 * 32 * 128 * 128;

    // Allocate and initialize host memory
    bf16 *matrix = (bf16 *)malloc(size * sizeof(bf16));
    const int N = 128;
    for (int idx = 0; idx < size; idx++) {
        int i = idx / N;
        int j = idx % N;
        float val = fmodf((i * 123 + j * 37) * 0.001f, 2.0f) - 1.0f;
        matrix[idx] = __float2bfloat16(val);
    }

    // Allocate device memory
    bf16 *d_src, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dest, size * sizeof(bf16)));
    CUDA_CHECK(
        cudaMemcpy(d_src, matrix, size * sizeof(bf16), cudaMemcpyHostToDevice));

    // Test TMA copy correctness
    printf("Testing TMA copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    launch_multiwarp_pipeline(d_dest, d_src, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *tma_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(tma_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool tma_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (tma_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));
            tma_correct = false;
            break;
        }
    }
    printf("TMA Copy: %s\n\n", tma_correct ? "PASSED" : "FAILED");
    free(tma_result);

    // Test simple copy correctness
    printf("Testing simple copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(d_dest, d_src,
                                                             size);
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *simple_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(simple_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool simple_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (simple_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));

            simple_correct = false;
            break;
        }
    }
    printf("Simple Copy: %s\n\n", simple_correct ? "PASSED" : "FAILED");
    free(simple_result);

    // Benchmark both kernels
    const int num_iters = 10;
    const size_t size_bytes = size * sizeof(bf16);

    if (tma_correct) {
        BENCHMARK_KERNEL((launch_multiwarp_pipeline(d_dest, d_src, size)),
                         num_iters, size_bytes, "TMA Copy");
    }

    if (simple_correct) {
        BENCHMARK_KERNEL(
            (simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(
                 d_dest, d_src, size),
             cudaDeviceSynchronize()),
            num_iters, size_bytes, "Simple Copy");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    free(matrix);
    return 0;
}