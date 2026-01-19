#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namcespace cg = cooperative_groups;

__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr int WARP_SIZE = 32;


// Kahan sum to reduce errors
// 1 thread per row
__global__ void sum_kernel_v1(const float *input, float *output, int M, int N) {
    const row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > N)
        return;
    
    input += row * N;
    float sum = 0.0f;
    float error = 0.0f;

    for (int col = 0; col < N; col++) {
        float item = input[col] - error;
        float new_sum = sum + item;
        error = new_sum - sum - item;
        sum = new_sum;
    }

    output[row] = sum;
}

void sum_v1(const float *input, float *output, int M, int N, int BLOCK_SIZE) {
    sum_kernel_v1<<<cdiv(M, BLOCK_SIZE), BLOCK_SIZE>>>(input, output, M, N);
}


// parallel sum with shared memory
// each thread block calculates sum for BLOCK_SIZE elements of input
__global__ void sum_kernel_v2(const float *input, float *output, int M, int N) {
    const int tid = threadIdx.x;
    const int BLOCK_SIZE = blockDim.x;
    const int col = blockIdx.x * BLOCK_SIZE + tid;
    const int row = blockIdx.y;

    extern __shared__ float shmem[];
    input += row * N;

    shmem[tid] = col < N ? input[col] : 0.0f;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    // global synchronization
    if (tid == 0)
        atomicAdd(output + row, shmem[0]);
}

void sum_v2(const float *input, float *output, int M, int N, int BLOCK_SIZE) {
    dim3 grid_size(cdiv(N, BLOCK_SIZE), M);
    const int shmem_size = sizeof(float) * BLOCK_SIZE;
    sum_kernel_v2<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N);
}


// thread-level reduction
// load a tile of BLOCK_SIZE ont at a time -> coalesced memory access
__device__ float thread_sum(const float *input, int TILE_SIZE, int BLOCK_SIZE, int tid, int max_idx) {
    float sum = 0.0f;
    for (int idx = tid; idx < TILE_SIZE; idx += BLOCK_SIZE)
        if (idx < max_idx)
            sum += input[idx];
    return sum;
}

// thread coarsening
// each thread block calculates sum for TILE_SIZE elements of input
__global__ void sum_kernel_v3(const float *input, float *output, int M, int N, int TILE_SIZE) {
    const int tid = threadIdx.x;
    const int BLOCK_SIZE = blockDim.x;
    const int tile_id = blockIdx.x;
    const int row = blockIdx.y;

    extern __shared__ float shmem[];
    input += row * N + tile_id * TILE_SIZE;

    shmem[tid] = thread_sum(input, TILE_SIZE, BLOCK_SIZE, tid, N - tile_id * TILE_SIZE);
    __syncthread();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthread();
    }

    if (tid == 0)
        atomAdd(output + row, shmem[0]);
}

void sum_v3(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE) {
    dim3 grid_size(cdiv(N, TILE_SIZE), M);
    const int shmem_size = sizeof(float) * BLOCK_SIZE;
    sum_kernel_v3<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}


// warp-level reduction
// NOTE: block_size must be >= 64 for this kernel
__global__ void sum_kernel_v4(const float *input, float *output, int M, int N, int TILE_SIZE) {
    const int tid = threadIdx.x;
    const int BLOCK_SIZE = blockDim.x;
    const int tile_id = blockIdx.x;
    const int row = blockIdx.y;

    extern __shared__ float shmem[];
    input += row * N + tile_id * TILE_SIZE;

    shmem[tid] = thread_sum(input, TILE_SIZE, BLOCK_SIZE, tid, N - tile_id * TILE_SIZE);
    __syncthreads();

    // block-level reduction
    // no warp divergence since all threads in a 32-thread warp will either do the addition or not.
    for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    // warp-level reduction
    float sum;
    if (tid < WARP_SIZE) {
        sum = shmem[tid];
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
        }
    }

    // grid-level reduction
    if (tid == 0)
        atomAdd(output + row, sum);
}

void sum_v4(const float *input, float *output, int M, int N, int TILE_SIZE, int BLOCK_SIZE) {
    assert((BLOCK_SIZE >= 64) && "block_size must be >= 64");
    dim3 grid_size(cdiv(N, TILE_SIZE), M);
    const int shmem_size = sizeof(float) * BLOCK_SIZE;
    sum_kernel_v4<<<grid_size, BLOCK_SIZE, shmem_size>>>(input, output, M, N, TILE_SIZE);
}
