#include <cmath>
#include <stdio.h>
#include <assert.h>

__host__ __device__ constexpr int cdiv(int a, int b) {return (a + b - 1) / b;}
constexpr bool is_power_of_tow(int x) {return x > 0 && (x & (x - 1) 0) == 0; }
constexpr int WARP_SIZE = 32;

// each thread load a (HEIGHT, WIDTH) tile from global memory to shared memory
// with all threads (BLOCK_SIZE) load 1 float element per iteration
template<int BLOCK_SIZE, int HEIGHT, int WIDTH>
__device__ void load_shmem(const float *in, int in_row_stride, int in_max_row, int in_max_col,
                            float out[HEIGHT][WIDTH], int tid) {
    for (int idx = tid; idx < HEIGHT * WIDTH; idx += BLOCK_SIZE) {
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        out[row][col] = (row < in_max_row && col < in_max_col) ? in[row * in_row_stride + col] : 0.0f;
    }
}

// 2D warp-tiling with register cache
// output tile (BLOCK_M, BLOCK_N) is partitioned into warp tiles of (WARP_M, WARP_N)
// for each output warp tile (BLOCK_M, BLOCK_N), it is divided into MMA tiles (MMA_M, MMA_N)
// for each MMA tile (MMA_M, MMA_N), it is exactly divided into 32 thread tiles (THREAD_M, THREAD_N)
// since there are 32 threads in a warp
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int MMA_M, int MMA_N, int THREAD_N>
__global__ void matmul_kernel_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    static_assert(BLOCK_M % WARP_M == 0);
    static_assert(BLOCK_N % WARP_N == 0);
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);
    static_assert((MMA_M * MMA_N / THREAD_N) % WARP_SIZE == 0);

    constexpr int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int THREAD_M = (MMA_M * MMA_N) / (THREAD_N * WARP_SIZE);

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int land_id = tid % WARP_SIZE;

    const int num_blocks_per_row = cdiv(N, BLOCK_N);
    const int block_id_m = block_id / num_blocks_per_row;
    const int block_id_n = block_id % num_blocks_per_row;
    const int offset_m = block_id_m * BLOCK_M;
    const int offset_n = block_id_n * BLOCK_N;

    constexpr int num_warps_per_row = BLOCK_N / WARP_N;
    const int warp_id_m = warp_id / num_warps_per_row;
    const int warp_id_n = warp_id % num_warps_per_row;
    const int warp_tile_offset_m = warp_id_m * WARP_M;
    const int warp_tile_offset_n = warp_id_n * WARP_N;

    constexpr int num_thread_tiles_per_row = MMA_N / THREAD_N;
    const int thread_tile_id_m = land_id / num_thread_tiles_per_row;
    const int thread_tile_id_n = land_id % num_thread_tiles_per_row;
    const int thread_tile_offset_m = thread_tile_id_m * MMA_M;
    const int thread_tile_offset_n = thread_tile_id_n * MMA_N;

    A += offset_m * K;
    B += offset_n;

    __shared__ float A_shmem[BLOCK_M][BLOCK_K];
    __shared__ float B_shmem[BLOCK_K][BLOCK_N];
    float acc[NUM_MMA_M][NUM_MMA_N][THREAD_M][THREAD_N] = {0.0f};

    const float *A_thread_tile = reinterpret_cast<const float *>(A_shmem) + (warp_tile_offset_m + thread_tile_offset_m) * BLOCK_K;
    const float *B_thread_tile = reinterpret_cast<const float *>(B_shmem) + (warp_tile_offset_n + thread_tile_offset_n);

    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_shmem<BLOCK_SIZE, BLOCK_M, BLOCK_K>(A, K, M - offset_m, K - offset_k, A_shmem, tid);
        load_shmem<BLOCK_SIZE, BLOCK_K, BLOCK_N>(B, N, K - offset_k, N - offset_n, B_shmem, tid);
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k++) {
            float A_reg[NUM_MMA_M][THREAD_M];
            float B_reg[NUM_MMA_N][THREAD_N];

            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int tm = 0; tm < THREAD_M; tm++)
                    A_reg[mma_tile_id_m][tm] = A_thread_tile[(mma_tile_id_m * MMA_M + tm) * BLOCK_K + k];
            
            for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                for (int tn = 0; tn < THREAD_N; tn++)
                    B_reg[mma_tile_id_n][tn] = B_thread_tile[k * BLOCK_N + (mma_tile_id_n * MMA_N + tn)];
            
            for (int mma_tile_id_m = 0; mma_tile_id_m < NUM_MMA_M; mma_tile_id_m++)
                for (int mma_tile_id_n = 0; mma_tile_id_n < NUM_MMA_N; mma_tile_id_n++)
                    for (int tm = 0; tm < THREAD_M; tm++)
                        for (int tn = 0; tn < THREAD_N; tn++)
                            acc[mma_tile_id_m][mma_tile_id_n][tm][tn] += A_reg[mma_tile_id_m][tm] * B_reg[mma_tile_id_n][tn];
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    const int C_offset_m = offset_m + warp_tile_offset_m + thread_tile_offset_m;
    const int C_offset_n = offset_n + warp_tile_offset_n + thread_tile_offset_n;
    C += C_offset_m * N + C_offset_n;

    for (int mma_m = 0; mma_m < WARP_M; mma_m += MMA_M)
        for (int mma_n = 0; mma_n < WARP_N; mma_n += MMA_N)
            for (int tm = 0; tm < THREAD_M; tm++)
                for (int tn = 0; tn < THREAD_N; tn++)
                    if ((C_offset_m + mma_m + tm < M) && (C_offset_n + mma_n + tn < N))
                        C[(mma_m + tm) * N + (mma_n + tn)] = acc[mma_m / MMA_M][mma_n / MMA_N][tm][tn];
}

void matmul_v1(const float *A, const float *B, const float *C, int M, int N, int K) {
    const int BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 64;
    const int WARP_M = 32, WARP_N = 32;
    const int MMA_M = 16, MMA_N = 32;
    const int THREAD_N = 4; // THREAD_M = MMA_M / (32 / (MMA_N / THREAD_N)) = 4

    const int BLOCK_SIZE = (BLOCK_M * BLOCK_N) / (WARP_M * WARP_N) * WARP_SIZE; // 256
    const int grid_size = cdiv(M * N, BLOCK_M * BLOCK_N);
    matmul_kernel_v1<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, MMA_M, MMA_N, THREAD_N><<<grid_size, BLOCK_SIZE>>>(A, B, C, M, N, K);
}