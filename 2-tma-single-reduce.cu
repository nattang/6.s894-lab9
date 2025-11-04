// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <stdio.h>

#include "tma-interface.cuh"

// Type alias for bfloat16
typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 2: Single Block, Single Tile TMA Reduce
////////////////////////////////////////////////////////////////////////////////

// Feel free to change the interface to this function if you
// are using a different tile dimension that 2d.
__device__ static __forceinline__ void
cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
    void *dstMem, void *srcMem, int size)
{
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[%0], [%1], %2;\n"
        :
        : "l"(dstMem), 
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(srcMem))),    
          "r"(size)
        : "memory");
}

template <int TILE_M, int TILE_N>
__global__ void
single_tma_reduce(__grid_constant__ const CUtensorMap src_map,
                  __grid_constant__ const CUtensorMap dest_map,
                  bf16 *dest)
{
    __shared__ bf16 shmem[TILE_M][TILE_N];
    __shared__ uint64_t bar;

    init_barrier(&bar, 1);
    async_proxy_fence();

    int expected_bytes = TILE_M * TILE_N * sizeof(bf16);
    expect_bytes_and_arrive(&bar, expected_bytes);

    cp_async_bulk_tensor_2d_global_to_shared(
        &shmem,   // void* smem_dest,
        &src_map, // const CUtensorMap* tensor_map,
        0, 0,     // int c0, int c1,
        &bar      // uint64_t* bar
    );

    wait(&bar, 0);

    size_t size = TILE_M * TILE_N * sizeof(bf16);
    cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
        dest,
        &shmem,
        size);

    tma_commit_group();
    tma_wait_until_pending<0>();

}

template <int TILE_M, int TILE_N>
void launch_single_tma_reduce(bf16 *src, bf16 *dest)
{
    CUtensorMap src_map;
    CUtensorMap dest_map;

    cuuint64_t dims[2] = {TILE_N, TILE_M};
    cuuint64_t strides[2] = {TILE_N * sizeof(bf16), sizeof(bf16)};
    cuuint32_t tile_dims[2] = {TILE_N, TILE_M};
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
    size_t shmem_size_bytes = TILE_M * TILE_N * sizeof(bf16) + sizeof(uint64_t);
    single_tma_reduce<TILE_M, TILE_N><<<1, 1, shmem_size_bytes>>>(src_map, dest_map, dest);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main()
{
    const int M = 64;
    const int N = 128;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    bf16 *matrix = (bf16 *)malloc(total_size * sizeof(bf16));
    bf16 *d_matrix;
    bf16 *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(bf16));
    cudaMalloc(&d_dest, total_size * sizeof(bf16));

    // Copy in 1s for the reduction.
    for (int i = 0; i < total_size; i++)
    {
        matrix[i] = 1;
    }
    cudaMemcpy(d_dest, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    // Initialize source matrix on host
    std::default_random_engine generator(0);
    std::normal_distribution<float> dist(0, 1);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = dist(generator);
            matrix[i * N + j] = __float2bfloat16(val);
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA reduce kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_reduce<M, N>(d_matrix, d_dest);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    bf16 *final_output = (bf16 *)malloc(total_size * sizeof(bf16));
    cudaMemcpy(final_output, d_dest, total_size * sizeof(bf16),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++)
    {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j] + 1.0f;
        float computed = (float)final_output[i * N + j];
        float diff = std::fabs(ref - computed);
        if (diff > 0.1)
        {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);
    free(final_output);

    return 0;
}