// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

using datatype = uint8_t;

////////////////////////////////////////////////////////////////////////////////
// Part 5: Reverse-Engineering TMA 64B Swizzle
////////////////////////////////////////////////////////////////////////////////

/// <--- your code here --->

template <int TILE_M, int TILE_N, int OFFSET>
__global__ void tma_swizzle(__grid_constant__ const CUtensorMap src_map,
                            datatype *dest) {
    /*
     * IMPORTANT REQUIREMENT FOR PART 5:
     *
     * To get credit, you need to use smem_buffer to store your TMA data.
     * Do not edit the setup for smem_buffer.
     */
    __shared__ alignas(128)
        datatype smem_buffer_abs[TILE_M * TILE_N + 128 * OFFSET];
    datatype *smem_buffer = &smem_buffer_abs[128 * OFFSET];

    __shared__ alignas(8) uint64_t mbar;

    int lane = threadIdx.x; 

    if (lane == 0) {
        init_barrier(&mbar, 1); 
        async_proxy_fence();
    }

    __syncthreads();

    if (lane == 0) {
        cp_async_bulk_tensor_2d_global_to_shared(smem_buffer, &src_map, 0, 0, &mbar);
        expect_bytes_and_arrive(&mbar, TILE_M * TILE_N * sizeof(datatype));
    }
    
    wait(&mbar, 0);

    for (int i = threadIdx.x; i < TILE_M; i+=blockDim.x) {
        for (int j = threadIdx.y; j < TILE_N; j+=blockDim.y) {
            uint64_t og_addr = reinterpret_cast<uint64_t>(&dest[i*TILE_N+j]);

            uint64_t swizzle_switch = ((reinterpret_cast<uint64_t>(smem_buffer) >> 7) & 0b11) << 4;

            datatype * new_addr = reinterpret_cast<datatype*>(og_addr ^ swizzle_switch);

            *new_addr = smem_buffer[i * TILE_N + j];
        }
    }



    // Cast to a "shared pointer" so that it works with
    // cp_async_bulk_tensor_2d_global_to_shared
    /* TODO: your launch code here... */
}

template <int TILE_M, int TILE_N, int OFFSET>
void launch_tma_swizzle(datatype *src, datatype *dest) {

    /*
     * IMPORTANT REQUIREMENT FOR PART 5:
     *
     * To get credit for this part, launch the tma_swizzle
     * kernel with the CU_TENSOR_MAP_SWIZZLE_64B setting.
     */

    CUtensorMap src_map;

    const cuuint64_t global_dim[2] = {TILE_N, TILE_M};
    const cuuint64_t global_strides[1] = {TILE_N * sizeof(datatype)};
    const cuuint32_t box_dim[2] = {TILE_N,TILE_M};
    const cuuint32_t element_strides[2] = {1,1};

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &src_map, 
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2, 
            src, 
            global_dim, 
            global_strides,
            box_dim, 
            element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );

    tma_swizzle<TILE_M, TILE_N, OFFSET><<<1,32>>>(src_map, dest);

    datatype x[TILE_M*TILE_N]; 

    CUDA_CHECK(cudaMemcpy(x, dest, TILE_M * TILE_N * sizeof(datatype), cudaMemcpyDeviceToHost));
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

template <int M, int N, int OFFSET>
void set_up_test(datatype *matrix, datatype *d_matrix, datatype *d_dest) {

    printf("Testing offset %d...\n", OFFSET);

    const uint64_t total_size = M * N;
    datatype *final_output = (datatype *)malloc(total_size * sizeof(datatype));
    // Zero out destination buffer
    for (int i = 0; i < total_size; i++) {
        final_output[i] = 0;
    }
    cudaMemcpy(d_dest, final_output, total_size * sizeof(datatype),
               cudaMemcpyHostToDevice);
    // Launch kernel
    launch_tma_swizzle<M, N, OFFSET>(d_matrix, d_dest);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    cudaMemcpy(final_output, d_dest, total_size * sizeof(datatype),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++) {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j];
        float computed = (float)final_output[i * N + j];
        if (ref != computed) {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    free(final_output);
}

template <int M, int N>
void run_test(datatype *matrix, datatype *d_matrix, datatype *d_dest) {
    // Test with different offsets
    set_up_test<M, N, 0>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 1>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 2>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 3>(matrix, d_matrix, d_dest);
}

int main() {
    const int M = 1;
    const int N = 64;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    datatype *matrix = (datatype *)malloc(total_size * sizeof(datatype));
    datatype *d_matrix;
    datatype *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(datatype));
    cudaMalloc(&d_dest, total_size * sizeof(datatype));

    // Initialize source matrix on host
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = i + j;
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(datatype),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA swizzle tests...\n\n");

    run_test<M, N>(matrix, d_matrix, d_dest);

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);

    return 0;
}
