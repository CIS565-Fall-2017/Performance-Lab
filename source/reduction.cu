#include "memBenchmark.h"
#include "termcolor.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// NVTX Dir: C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt
#include <nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

// Number of element to reduce
static const int n_elements = 8 * 1024 * 1024;

// Number of threads per block to use for all kernels
static const int threads = 256;

struct DIMS1D
{
    int dimThreads;
    int dimBlocks;
};

#define CUDA(call) do {                                 \
    cudaError_t e = (call);                             \
    if (e == cudaSuccess) break;                        \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",           \
            __LINE__, cudaGetErrorString(e), e);        \
    exit(1);                                            \
} while (0)

inline unsigned divup(unsigned n, unsigned div)
{
    return (n + div - 1) / div;
}

void printResults(double timeInMilliseconds, int iterations)
{
    // print out the time required for the kernel to finish the transpose operation
    double bandwidth = (iterations * 1e-9 * (double)(n_elements * sizeof(float)))
                       / (timeInMilliseconds * 1e-3);
    std::cout << "Elapsed Time for " << iterations << " runs = " << round(timeInMilliseconds) << "ms" << std::endl;
    std::cout << termcolor::bold << termcolor::red << termcolor::on_white
              << "Bandwidth (GB/s) = " << std::setprecision(4) << bandwidth
              << termcolor::reset << std::endl;
    std::cout.clear();
}

// Check errors
bool postprocess(const float *ref, const float *res, int n)
{
    bool passed = true;
    for(int i = 0; i < n; i++)
    {
        if (std::abs(res[i] - ref[i]) / n_elements > 1e-6)
        {
            std::cout.precision(6);
            std::cout << "ID: " << i << " \t Res: " << res[i] << " \t Ref: " << ref[i] << std::endl;
            std::cout << termcolor::blink << termcolor::white << termcolor::on_red << "*** FAILED ***" << termcolor::reset << std::endl;
            passed = false;
            break;
        }
    }
    if(passed)
        std::cout << termcolor::green << "Post process check passed!!" << termcolor::reset << std::endl;

    return passed;
}

static float reduce_cpu(const float *data, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 0: Interleaved Addressing
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage0(const float* d_idata, float* d_odata, int n)
{
    // Dynamic allocation of shared memory - See kernel call in host code
    extern __shared__ float smem[];

    // Calculate 1D Index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy input data to shared memory
    // Note: Use block index for shared memory
    // Also check for bounds
    if(idx < n)
        smem[threadIdx.x] = d_idata[idx];

    // Where do we need to put __syncthreads()? Do we need it at all?
    __syncthreads();

    // Reduce within block
    // Start from c = 1, upto block size, each time doubling the offset
    for(int c = 1; c < blockDim.x; c *= 2)
    {
        // Add only on left index of each level
        if (threadIdx.x % (2 * c) == 0)
            smem[threadIdx.x] += smem[threadIdx.x + c];

        __syncthreads();
    }

    // Copy result of reduction to global memory
    // Which index of d_odata do we write to?
    // In which index of smem is the result stored?
    // Do we need another syncthreads before writing to global memory?
    // Use only one thread to write to global memory
    if(threadIdx.x == 0)
        d_odata[blockIdx.x] = smem[0];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 1: Non-divergent Addressing
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
//
// The only difference between stage0 and stage1 is the reduction for loop
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage1(const float* d_idata, float* d_odata, int n)
{
    // Allocate dynamic shared memory, Calculate 1D Index and Copy input data into shared memory
    // Exactly same as reduce_stage0

    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
        smem[threadIdx.x] = d_idata[idx];

    __syncthreads();

    // This is the part that differes from reduce_stage0
    // Reduce within block with coalesced indexing pattern
    // Change the for-loop to use indexing that reduces warp-divergence
    for(int c = 1; c < blockDim.x; c *= 2)
    {
        int index = 2 * c * threadIdx.x;
        if(index < blockDim.x)
            smem[index] += smem[index + c];

        __syncthreads();
    }

    // Copy result of reduction to global memory - Same as reduce_stage0
    if(threadIdx.x == 0)
        d_odata[blockIdx.x] = smem[0];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 2: Warp Management without Bank Conflicts
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
//
// The only difference between stage1 and stage2 is the reduction for loop
// This time, we reduce start from blockDim.x and divide by 2 in each iteration
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage2(const float* d_idata, float* d_odata, int n)
{
    // Allocate dynamic shared memory, Calculate 1D Index and Copy input data into shared memory
    // Exactly same as reduce_stage1

    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
        smem[threadIdx.x] = d_idata[idx];

    __syncthreads();

    // This is the part that differes from reduce_stage1
    // Reduce within block with coalesced indexing pattern and avoid bank conflicts
    // Change the for-loop to use indexing that reduces warp-divergence
    // Start from blockDim.x / 2 and divide by 2 until we hit 1
    for(int c = blockDim.x / 2; c > 0; c >>= 1) /// Do sum in log(N) iterations [c>>=1 = c/=2]
    {
        // Inside of the loop is the similar to reduce_stage0 (not reduce_stage1)
        // The difference is in the if condition
        if(threadIdx.x < c)
            smem[threadIdx.x] += smem[threadIdx.x + c];

        __syncthreads();
    }

    // Copy result of reduction to global memory - Same as reduce_stage1
    if(threadIdx.x == 0)
        d_odata[blockIdx.x] = smem[0];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 3: Add During Load, Use tile to reduce number of blocks
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage3_TILE  : Tiling factor
//
// In this kernel, we will add on load when copying data into shared memory
// The difference between stage3 and stage2 is how we load data into shared memory
// Each block does work of stage3_TILE * blockDim.x elements
////////////////////////////////////////////////////////////////////////////////
const int stage3_TILE = 2;
__global__ void reduce_stage3(const float* d_idata, float* d_odata, int n)
{
    // Allocate dynamic shared memory

    extern __shared__ float smem[];

    // Calculate 1D index similar to stage2, but multiply by stage3_TILE
    int idx = blockIdx.x * blockDim.x * stage3_TILE + threadIdx.x;

    // Copy input data to shared memory. Add on load.
    if(idx < n)
    {
        smem[threadIdx.x] = 0;
        for(int c = 0; c < stage3_TILE; c++)
        {
            //Copy and add block data into shared memory
            if(idx + c * blockDim.x < n)
                smem[threadIdx.x] += d_idata[idx + c * blockDim.x];
        }
    }

    __syncthreads();

    // Reduce the block same as reduce_stage2
    for(int c = blockDim.x / 2; c > 0; c >>= 1)
    {
        if(threadIdx.x < c)
            smem[threadIdx.x] += smem[threadIdx.x + c];

        __syncthreads();
    }

    //Copy result of reduction to global memory - Same as reduce_stage2
    if(threadIdx.x == 0)
        d_odata[blockIdx.x] = smem[0];
}

// warpReduce function for reduce_stag4 that reduces 2 warps into one value
__device__ void warpReduce(volatile float* smem, int tid)
{
    //Write code for warp reduce here
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8 ];
    smem[tid] += smem[tid + 4 ];
    smem[tid] += smem[tid + 2 ];
    smem[tid] += smem[tid + 1 ];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 4: Warp Loop Unrolling
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage4_TILE  : Tiling factor - How does tuning this change performance?
//
// The reduce_stage4 kernel improves on reduce_stage3 by unrolling the block
// reduction by unrolling the loop that operates within a warp.
// Each block does work of stage4_TILE * blockDim.x elements
//
// This kernel also uses the warpReduce device function above
////////////////////////////////////////////////////////////////////////////////
const int stage4_TILE = 2;
__global__ void reduce_stage4(const float* d_idata, float* d_odata, int n)
{
    // Allocate dynamic shared memory, Calculate 1D Index and
    // Copy input data and add on load into shared memory
    // Exactly same as reduce_stage3. Use stage4_TILE instead of stage3_TILE.

    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x * stage4_TILE + threadIdx.x;

    if(idx < n)
    {
        smem[threadIdx.x] = 0;
        for(int c = 0; c < stage4_TILE; c++)
        {
            if(idx + c * blockDim.x < n)
                smem[threadIdx.x] += d_idata[idx + c * blockDim.x];
        }
    }

    __syncthreads();

    // Reduce within block with coalesced indexing pattern and avoid bank conflicts
    // Split the block reduction into 2 parts.
    // Part 1 is the same as reduce stage3, but only for c > 32
    for(int c = blockDim.x / 2; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
            smem[threadIdx.x] += smem[threadIdx.x + c];

        __syncthreads();
    }

    // Part 2 then uses the warpReduce function to reduce the 2 warps
    // The reason we stop the previous loop at c > 32 is because
    // warpReduce can reduce 2 warps only 1 warp
    if(threadIdx.x < 32)
        warpReduce(smem, threadIdx.x);

    // Copy result of reduction to global memory - Same as reduce_stage3
    if(threadIdx.x == 0)
        d_odata[blockIdx.x] = smem[0];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 5: Completely unrolled blocks using templates
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage5_TILE  : Tiling factor - How does tuning this change performance?
//
// The reduce_stage5 kernel is the same as reduce_stage4 except part 1 of block reduction
// We simply unroll the entire for loop into individual statements wrapper by if conditions
// Why do we need to use templates? How do they improve performance?
// Each block does work of stage5_TILE * blockDim.x elements
//
// This kernel also uses the warpReduce device function above
////////////////////////////////////////////////////////////////////////////////
const int stage5_TILE = 2;
template<unsigned int blockSize>
__global__ void reduce_stage5(const float* d_idata, float* d_odata, int n)
{
    // Allocate dynamic shared memory, Calculate 1D Index and
    // Copy input data and add on load into shared memory
    // Exactly same as reduce_stage4. Use stage5_TILE instead of stage4_TILE.
    // Use #pragma unroll around the load loop

    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x * stage5_TILE + threadIdx.x;

    // Store the threadIdx.x in a register
    int tid = threadIdx.x;
    if(idx < n)
    {
        smem[tid] = 0;
        // We need the loop to only use statically assigned variables. Otherwise the loop can't unroll.
        #pragma unroll
        for(int i = 0; i < blockSize * stage5_TILE; i += blockSize)
        {
            if(idx + i < n)
                smem[tid] += d_idata[idx + i];
        }
    }

    __syncthreads();

    // Reduce the block using the same part1 and part2 split that we used in reduce_stage4
    // Except, here write explicit statements instead of the for loops
    if(blockSize >= 1024) {
        if(threadIdx.x < 512)   { smem[tid] += smem[tid + 512]; __syncthreads(); }
    }
    if(blockSize >= 512) {
        if(threadIdx.x < 256)   { smem[tid] += smem[tid + 256]; __syncthreads(); }
    }
    if(blockSize >= 256) {
        if(threadIdx.x < 128)   { smem[tid] += smem[tid + 128]; __syncthreads(); }
    }
    if(blockSize >= 128) {
        if(threadIdx.x < 64)    { smem[tid] += smem[tid + 64];  __syncthreads(); }
    }

    // Part 2 is the same as reduce_stage4
    if(tid < 32)
        warpReduce(smem, tid);

    // Copy result of reduction to global memory - Same as reduce_stage4
    if(tid == 0)
        d_odata[blockIdx.x] = smem[0];
}

// Wrapper for reduce_stage0 - Allows recursive kernel calls for reduction
float reduce_stage0_wrapper(const float *d_idata, float *d_odata, const int elements)
{
    //Calculate threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks = divup(elements, dims.dimThreads);

    // If number of elements is less than 1 block, then do CPU reduce
    // Otherwise recursively call the wrapper
    if (elements < dims.dimThreads) {
        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(elements * sizeof(float));
        CUDA(cudaMemcpy(h_blocks, d_odata, elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;

        for(int i = 0; i < elements; i++)
            gpu_result += h_blocks[i];

        free(h_blocks);

        return gpu_result;
    } else {
        reduce_stage0<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, elements);

        return reduce_stage0_wrapper(d_odata, d_odata, dims.dimBlocks);
    }
}

// Wrapper for reduce_stage1 - Allows recursive kernel calls for reduction
float reduce_stage1_wrapper(const float *d_idata, float *d_odata, const int elements)
{
    //Calculate threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks = divup(elements, dims.dimThreads);

    // If number of elements is less than 1 block, then do CPU reduce
    // Otherwise recursively call the wrapper
    if (elements < dims.dimThreads) {
        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(elements * sizeof(float));
        CUDA(cudaMemcpy(h_blocks, d_odata, elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;

        for(int i = 0; i < elements; i++)
            gpu_result += h_blocks[i];

        free(h_blocks);

        return gpu_result;
    } else {
        reduce_stage1<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, elements);

        return reduce_stage1_wrapper(d_odata, d_odata, dims.dimBlocks);
    }
}

// Wrapper for reduce_stage2 - Allows recursive kernel calls for reduction
float reduce_stage2_wrapper(const float *d_idata, float *d_odata, const int elements)
{
    //Calculate threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks = divup(elements, dims.dimThreads);

    // If number of elements is less than 1 block, then do CPU reduce
    // Otherwise recursively call the wrapper
    if (elements < dims.dimThreads) {
        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(elements * sizeof(float));
        CUDA(cudaMemcpy(h_blocks, d_odata, elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;

        for(int i = 0; i < elements; i++)
            gpu_result += h_blocks[i];

        free(h_blocks);

        return gpu_result;
    } else {
        reduce_stage2<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, elements);

        return reduce_stage2_wrapper(d_odata, d_odata, dims.dimBlocks);
    }
}

// Wrapper for reduce_stage3 - Allows recursive kernel calls for reduction
float reduce_stage3_wrapper(const float *d_idata, float *d_odata, const int elements)
{
    //Calculate threads per block and total blocks required - Remember to use stage3_TILE
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks = divup(elements, dims.dimThreads * stage3_TILE);

    // If number of elements is less than 1 block, then do CPU reduce
    // Otherwise recursively call the wrapper
    if (elements < dims.dimThreads * stage3_TILE) {
        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(elements * sizeof(float));
        CUDA(cudaMemcpy(h_blocks, d_odata, elements * sizeof(float), cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;

        for(int i = 0; i < elements; i++)
            gpu_result += h_blocks[i];

        free(h_blocks);

        return gpu_result;
    } else {
        reduce_stage3<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, elements);

        return reduce_stage3_wrapper(d_odata, d_odata, dims.dimBlocks);
    }
}

int main()
{
    // Calculate bytes needed for input
    const unsigned bytes = n_elements * sizeof(float);

    // Allocate memory and initialize elements
    // Let's use pinned memory for host
    float *h_idata;
    CUDA(cudaMallocHost((void**)&h_idata, bytes));

    // Fill random values into the host array
    {
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        std::uniform_real_distribution<float> distribution(-1, 1);

        for (int i = 0; i < n_elements; i++) {
            h_idata[i] = distribution(generator);
        }
    }

    // Copy input data into device memory
    float *d_idata = NULL;
    CUDA(cudaMalloc((void **)&d_idata, bytes));
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Compute Gold Standard using CPU
    const float gold_result = reduce_cpu(h_idata, n_elements);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***CPU Reduce***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("CPU Reduce");

        float cpu_result = 0;

        int iterations = 100;

        // start the timer
        Timer hTimer;
        nvtxRangeId_t rangeBenchmark = nvtxRangeStart("CPU Reduce Benchmark");
        for(int k = 0; k < iterations; k++)
        {
            cpu_result = reduce_cpu(h_idata, n_elements);
        }
        nvtxRangeEnd(rangeBenchmark);

        // stop the timer
        double time = hTimer.elapsed() * 1000; //ms

        if(postprocess(&cpu_result, &gold_result, 1))
            printResults(time, iterations);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 0***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 0");

        // Calculate initial blocks
        int blocks = divup(n_elements, threads);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = blocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the wrapper
        float gpu_result = reduce_stage0_wrapper(d_idata, d_odata, n_elements);

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 0 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));

            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage0_wrapper(d_idata, d_odata, n_elements);
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 1***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 1");

        // Calculate initial blocks
        int blocks = divup(n_elements, threads);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = blocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the wrapper
        float gpu_result = reduce_stage1_wrapper(d_idata, d_odata, n_elements);

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 1 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));

            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage1_wrapper(d_idata, d_odata, n_elements);
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 2***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 2");

        // Calculate initial blocks
        int blocks = divup(n_elements, threads);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = blocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the wrapper
        float gpu_result = reduce_stage2_wrapper(d_idata, d_odata, n_elements);

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 2 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));

            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage2_wrapper(d_idata, d_odata, n_elements);
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 3***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 3");

        // Calculate initial blocks
        int blocks = divup(n_elements, threads * stage3_TILE);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = blocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the wrapper
        float gpu_result = reduce_stage3_wrapper(d_idata, d_odata, n_elements);

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 3 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));

            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage3_wrapper(d_idata, d_odata, n_elements);
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 4***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 4");

        // Calculate Threads per block and total blocks required
        // Use stage4_TILE in your grid calculation
        DIMS1D dims;
        dims.dimThreads = threads;
        dims.dimBlocks  = divup(n_elements, dims.dimThreads * stage4_TILE);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = dims.dimBlocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the kernel. Allocate dynamic shared memory
        reduce_stage4<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, n_elements);

        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(block_bytes);
        CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;
        for(int i = 0; i < dims.dimBlocks; i++)
            gpu_result += h_blocks[i];

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 4 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));


            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage4<<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, n_elements);

                cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

                for(int i = 0; i < dims.dimBlocks; i++)
                    gpu_result += h_blocks[i];
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        free(h_blocks);
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "******************************************" << std::endl;
    std::cout << "***Reduction Stage 5***" << std::endl;
    {
        nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 5");

        // Calculate Threads per block and total blocks required
        // Use stage5_TILE in your grid calculation
        DIMS1D dims;
        dims.dimThreads = threads;
        dims.dimBlocks  = divup(n_elements, dims.dimThreads * stage5_TILE);

        // Copy input data to device
        CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

        // Calculate bytes needed for output
        size_t block_bytes = dims.dimBlocks * sizeof(float);

        // Allocate memory for output on device
        float *d_odata = NULL;
        CUDA(cudaMalloc((void**)&d_odata, block_bytes));
        CUDA(cudaMemset(d_odata, 0, block_bytes));

        // Call the kernel. Allocate dynamic shared memory
        // Don't forget to add the template
        reduce_stage5<threads><<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, n_elements);

        // Copy result of block reduce to CPU and run CPU reduce
        float *h_blocks = (float *)malloc(block_bytes);
        CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

        // Secondary reduce on CPU
        float gpu_result = 0;
        for(int i = 0; i < dims.dimBlocks; i++)
            gpu_result += h_blocks[i];

        // Check the result and then run the benchmark.
        if(postprocess(&gpu_result, &gold_result, 1))
        {
            nvtxRangeId_t rangeBenchmark = nvtxRangeStart("Reduction Stage 5 Benchmark");

            //Start Benchmark
            int iterations = 100;
            CUDA(cudaEventRecord(start, 0));

            // Run multiple times for a good benchmark
            for(int i = 0; i < iterations; i++)
            {
                reduce_stage5<threads><<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, n_elements);

                cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

                for(int i = 0; i < dims.dimBlocks; i++)
                    gpu_result += h_blocks[i];
            }

            CUDA(cudaEventRecord(stop, 0));
            CUDA(cudaEventSynchronize(stop));

            nvtxRangeEnd(rangeBenchmark);

            float time_ms;
            CUDA(cudaEventElapsedTime(&time_ms, start, stop));

            printResults(time_ms, iterations);
        }

        // Cleanup
        free(h_blocks);
        cudaFree(d_odata);

        nvtxRangeEnd(range);
    }
    std::cout << "******************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    // Cleanup
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(stop));

    CUDA(cudaFreeHost(h_idata));
    CUDA(cudaFree(d_idata));

    return 0;
}
