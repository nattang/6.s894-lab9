// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// #define NUM_WARPS_PER_BLOCK 4
// #define ACTIVE_THREADS 4
// #define NUM_TILES_PER_THREAD 4

// #define BLOCK_TILE_DIM 256 // TODO: change this name
#define THREADS_PER_WARP 32
#define THREAD_TILE_DIM 32
#define WARPS_PER_BLOCK 2
#define PRODUCER_WARPS WARPS_PER_BLOCK / 2
#define TILES_PER_THREAD 1

#define BLOCK_TILE_X (THREAD_TILE_DIM * THREADS_PER_WARP)                   // 32 * 32 = 1024
#define BLOCK_TILE_Y (THREAD_TILE_DIM * WARPS_PER_BLOCK * TILES_PER_THREAD) // 32 * 2 * 4 = 256

// number of warps = block tile space / thread tile space / threads per warp

////////////////////////////////////////////////////////////////////////////////
// Part 4: Bring Your Own Warp Scheduler
////////////////////////////////////////////////////////////////////////////////

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int N)
{
    extern __shared__ __align__(128) unsigned char shmem_raw[];

    // size_t tile_bytes = THREAD_TILE_DIM * THREAD_TILE_DIM * THREADS_PER_WARP * PRODUCER_WARPS * sizeof(bf16);
    // bf16 *tile0 = reinterpret_cast<bf16 *>(shmem_raw);
    // bf16 *tile1 = reinterpret_cast<bf16 *>(reinterpret_cast<unsigned char *>(tile0) + tile_bytes);

    // // printf("tile_bytes = %lu\n", (unsigned long)tile_bytes);

    // size_t mbar_off = (tile_bytes + 8) & ~size_t(8);
    // uint64_t *pmbar0 = reinterpret_cast<uint64_t *>(reinterpret_cast<unsigned char *>(tile1) + mbar_off);
    // uint64_t *pmbar1 = pmbar0 + 1;
    // uint64_t *cmbar0 = pmbar1 + 1;
    // uint64_t *cmbar1 = cmbar0 + 1;

    bf16 *tile0 = reinterpret_cast<bf16 *>(shmem_raw);
    bf16 *tile1 = tile0 + (THREADS_PER_WARP * THREAD_TILE_DIM * THREAD_TILE_DIM * PRODUCER_WARPS);

    size_t tile_bytes = 2 * size_t(THREADS_PER_WARP) * THREAD_TILE_DIM * THREAD_TILE_DIM * PRODUCER_WARPS * sizeof(bf16);
    size_t mbar_off = (tile_bytes + 8) & ~size_t(8);
    uint64_t *pmbar0 = reinterpret_cast<uint64_t *>(shmem_raw + mbar_off);
    uint64_t *pmbar1 = pmbar0 + 1;
    uint64_t *cmbar0 = pmbar1 + 1;
    uint64_t *cmbar1 = cmbar0 + 1;

    int block_row = blockIdx.x * BLOCK_TILE_Y;
    int block_col = 0;

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int producer_id = warp_id % PRODUCER_WARPS;

    // int shmem_offset = thread_id * THREAD_TILE_DIM * THREAD_TILE_DIM;

    int tile_row_base = producer_id * THREAD_TILE_DIM * TILES_PER_THREAD;
    int tile_col_base = lane * THREAD_TILE_DIM;

    if (threadIdx.x == 0)
    {
        // init_barrier(&bar, THREADS_PER_WARP * WARPS_PER_BLOCK);
        init_barrier(pmbar0, THREADS_PER_WARP);
        init_barrier(pmbar1, THREADS_PER_WARP);
        init_barrier(cmbar0, THREADS_PER_WARP);
        init_barrier(cmbar1, THREADS_PER_WARP);
        async_proxy_fence();
    }
    __syncthreads();

    for (int t = 0; t < TILES_PER_THREAD; t++)
    {
        uint64_t *cur_pmbar = t & 1 ? pmbar1 : pmbar0;
        uint64_t *cur_cmbar = t & 1 ? cmbar1 : cmbar0;

        if (warp_id == 0)
        {
            if (t > 1)
            {
                wait(cur_pmbar, ((t / 2) + 1) % 2);
                wait(cur_cmbar, ((t / 2) + 1) % 2);
            }
            if (lane == 0 && blockIdx.x == 0)
                printf("block_row=%d, tile_row_base=%d, t*THREAD_TILE_DIM=%d, block_col=%d, tile_col_base=%d\n",
                       block_row, tile_row_base, t * THREAD_TILE_DIM, block_col, tile_col_base);

            // issue TMA load for tile (tile_row, tile_col)
            int expected_bytes = THREAD_TILE_DIM * THREAD_TILE_DIM * sizeof(bf16);
            expect_bytes_and_arrive(cur_pmbar, expected_bytes);
            __syncwarp();

            bf16 *base_tile_addr = t & 1 ? tile1 : tile0;

            cp_async_bulk_tensor_2d_global_to_shared(
                &base_tile_addr[lane * THREAD_TILE_DIM * THREAD_TILE_DIM],
                &tensor_map, // const CUtensorMap* tensor_map,
                block_row + tile_row_base + (t * THREAD_TILE_DIM),
                block_col + tile_col_base, // int c0, int c1,
                cur_pmbar                  // uint64_t* bar
            );
        }
        else if (warp_id == 1)
        {
            if (t > 1)
            {
                wait(cur_cmbar, ((t / 2) + 1) % 2);
            }

            if (lane == 0 && blockIdx.x == 0)
                printf("t=%d write buf=%d read buf=%d\n", t, (t & 1), (((t / 2)) % 2));

            wait(cur_pmbar, (t / 2) % 2);

            bf16 *base_tile_addr = t & 1 ? tile1 : tile0;

            // print first 4 values of the tile being written back
            // if (lane == 0 && blockIdx.x == 0)
            // {
            //     // printf("Tile to write back at t=%d:\n", t);

            //     for (int i = 0; i < 4; i++)
            //     {
            //         bf16 val = base_tile_addr[lane * THREAD_TILE_DIM * THREAD_TILE_DIM + i];
            //         printf("%.4f ", __bfloat162float(val));
            //     }
            //     printf("\n");
            // }

            cp_async_bulk_tensor_2d_shared_to_global(
                &dest_tensor_map, // const CUtensorMap* tensor_map,
                block_row + tile_row_base + (t * THREAD_TILE_DIM),
                block_col + tile_col_base,
                &base_tile_addr[lane * THREAD_TILE_DIM * THREAD_TILE_DIM] //   const void* smem_src
            );

            if (lane == 0 && blockIdx.x == 0)
            {
                printf("Tile to write back at t=%d:\n", t);
                printf("block_row=%d, tile_row_base=%d, t*THREAD_TILE_DIM=%d, block_col=%d, tile_col_base=%d\n",
                       block_row, tile_row_base, t * THREAD_TILE_DIM, block_col, tile_col_base);

                for (int i = 0; i < 10; i++)
                {
                    bf16 val = base_tile_addr[lane * THREAD_TILE_DIM * THREAD_TILE_DIM + i];
                    printf("%.4f ", __bfloat162float(val));
                }
                printf("\n");
            }

            // if (lane == 0)
            // {
            //     tma_commit_group();
            // }
            // __syncwarp();
            // tma_wait_until_pending<0>();
            // arrive(cur_cmbar, 1);
            // __syncwarp();
        }
    }
}

void launch_multiwarp_pipeline(bf16 *dest, bf16 *src, const int N)
{
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

    /* TODO: your launch code here... */
    CUtensorMap src_map;
    CUtensorMap dest_map;

    cuuint64_t map_y = N / BLOCK_TILE_X;
    cuuint64_t map_x = BLOCK_TILE_X;

    cuuint64_t dims[2] = {map_y, map_x};
    cuuint64_t strides[2] = {map_y * sizeof(bf16), sizeof(bf16)};
    cuuint32_t tile_dims[2] = {THREAD_TILE_DIM, THREAD_TILE_DIM};
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult src_descriptor = cuTensorMapEncodeTiled(
        &src_map,                           // CUtensorMap* tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // CUtensorMapDataType tensorDataType,
        2,                                  // cuuint32_t tensorRank,
        (void *)src,                        // void* globalAddress,
        dims,                               // const cuuint64_t* globalDim,
        strides,                            // const cuuint64_t* globalStrides,
        tile_dims,                          // const cuuint32_t* boxDim,
        elem_strides,                       // const cuuint32_t* elementStrides,
        {CU_TENSOR_MAP_INTERLEAVE_NONE},    // CUtensorMapInterleave interleave,
        {CU_TENSOR_MAP_SWIZZLE_NONE},       // CUtensorMapSwizzle swizzle,
        {CU_TENSOR_MAP_L2_PROMOTION_NONE},  // CUtensorMapL2promotion l2Promotion,
        {CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE} // CUtensorMapFloatOOBfill oobFill
    );

    CUresult dest_descriptor = cuTensorMapEncodeTiled(
        &dest_map,                          // CUtensorMap* tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,   // CUtensorMapDataType tensorDataType,
        2,                                  // cuuint32_t tensorRank,
        (void *)dest,                       // void* globalAddress,
        dims,                               // const cuuint64_t* globalDim,
        strides,                            // const cuuint64_t* globalStrides,
        tile_dims,                          // const cuuint32_t* boxDim,
        elem_strides,                       // const cuuint32_t* elementStrides,
        {CU_TENSOR_MAP_INTERLEAVE_NONE},    // CUtensorMapInterleave interleave,
        {CU_TENSOR_MAP_SWIZZLE_NONE},       // CUtensorMapSwizzle swizzle,
        {CU_TENSOR_MAP_L2_PROMOTION_NONE},  // CUtensorMapL2promotion l2Promotion,
        {CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE} // CUtensorMapFloatOOBfill oobFill
    );

    CUDA_CHECK(src_descriptor);
    CUDA_CHECK(dest_descriptor);

    // size_t shmem_size_bytes =
    //     (BLOCK_TILE_DIM)*NUM_WARPS_PER_BLOCK * BLOCK_TILE_DIM * sizeof(bf16) + sizeof(uint64_t);
    size_t shmem_size_bytes = 227 * 1000; // max
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_multiwarp_pipeline, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));

    // printf("shmem_size_bytes = %zu\n", shmem_size_bytes);

    int num_blocks = CEIL_DIV(N, BLOCK_TILE_Y * BLOCK_TILE_X);
    // printf("Launching %d blocks for TMA copy\n", num_blocks);
    tma_multiwarp_pipeline<<<num_blocks, WARPS_PER_BLOCK * THREADS_PER_WARP, shmem_size_bytes>>>(
        src_map, dest_map, N);

    // print first 16 values of dest:
    // printf("First 16 values of dest after TMA copy:\n");
    // bf16 host_dest[16];
    // CUDA_CHECK(cudaMemcpy(host_dest, dest, 16 * sizeof(bf16),
    //                       cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 16; i++)
    // {
    //     printf("%.4f ", __bfloat162float(host_dest[i]));
    // }
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

const int elem_per_block = 16384;
__global__ void simple_vector_copy(bf16 *__restrict__ dest,
                                   const bf16 *__restrict__ src, int N)
{
    constexpr int VEC_ELEMS = 8;
    using VecT = uint4;

    int total_vecs = elem_per_block / VEC_ELEMS;
    int start_vec = (blockIdx.x * blockDim.x) * total_vecs;

    const VecT *src_vec = reinterpret_cast<const VecT *>(src);
    VecT *dest_vec = reinterpret_cast<VecT *>(dest);

    for (int i = threadIdx.x; i < blockDim.x * total_vecs; i += blockDim.x)
    {
        dest_vec[start_vec + i] = src_vec[start_vec + i];
    }
}

#define BENCHMARK_KERNEL(kernel_call, num_iters, size_bytes, label)       \
    do                                                                    \
    {                                                                     \
        cudaEvent_t start, stop;                                          \
        CUDA_CHECK(cudaEventCreate(&start));                              \
        CUDA_CHECK(cudaEventCreate(&stop));                               \
        CUDA_CHECK(cudaEventRecord(start));                               \
        for (int i = 0; i < num_iters; i++)                               \
        {                                                                 \
            kernel_call;                                                  \
        }                                                                 \
        CUDA_CHECK(cudaEventRecord(stop));                                \
        CUDA_CHECK(cudaEventSynchronize(stop));                           \
        float elapsed_time;                                               \
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));     \
        float time_per_iter = elapsed_time / num_iters;                   \
        float bandwidth_gb_s = (2.0 * size_bytes * 1e-6 / time_per_iter); \
        printf("%s - Time: %.4f ms, Bandwidth: %.2f GB/s\n", label,       \
               time_per_iter, bandwidth_gb_s);                            \
        CUDA_CHECK(cudaEventDestroy(start));                              \
        CUDA_CHECK(cudaEventDestroy(stop));                               \
    } while (0)

int main()
{
    const size_t size = 132 * 10 * 32 * 128 * 128;

    // Allocate and initialize host memory
    bf16 *matrix = (bf16 *)malloc(size * sizeof(bf16));
    const int N = 128;
    for (int idx = 0; idx < size; idx++)
    {
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
    int num_incorrect = 0;
    for (int idx = 0; idx < size; idx++)
    {
        if (tma_result[idx] != matrix[idx])
        {
            printf("mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));
            tma_correct = false;
            // break;
            num_incorrect++;
            if (num_incorrect > 10)
            {
                break;
            }
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
    for (int idx = 0; idx < size; idx++)
    {
        if (simple_result[idx] != matrix[idx])
        {
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

    if (tma_correct)
    {
        BENCHMARK_KERNEL((launch_multiwarp_pipeline(d_dest, d_src, size)),
                         num_iters, size_bytes, "TMA Copy");
    }

    if (simple_correct)
    {
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