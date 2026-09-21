/*
 * 3D DCT CUDA Kernel
 * 实现真正的 CUDA Tiles 加速的 3D DCT 变换
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Tile 大小配置
#define TILE_D 8
#define TILE_H 8
#define TILE_W 8
#define TILE_C 16

// DCT 预计算系数
__constant__ float dct_coeff[TILE_D][TILE_D];
__constant__ float idct_coeff[TILE_D][TILE_D];

// 初始化 DCT 系数
__global__ void init_dct_coefficients() {
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    if (i < TILE_D && j < TILE_D) {
        float alpha_i = (i == 0) ? sqrtf(1.0f / TILE_D) : sqrtf(2.0f / TILE_D);
        float alpha_j = (j == 0) ? sqrtf(1.0f / TILE_D) : sqrtf(2.0f / TILE_D);
        dct_coeff[i][j] = alpha_i * alpha_j * cosf(M_PI * i * (2.0f * j + 1.0f) / (2.0f * TILE_D));
        idct_coeff[i][j] = alpha_i * alpha_j * cosf(M_PI * j * (2.0f * i + 1.0f) / (2.0f * TILE_D));
    }
}

// 1D DCT 变换（沿 Z 轴）
__global__ void dct1d_z_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    // 计算全局索引
    int c = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || h >= height || w >= width) return;
    
    // Shared Memory 缓存当前列的数据
    __shared__ float shared_data[TILE_D];
    
    // 计算当前列的起始位置
    int col_idx = c * depth * height * width + h * width + w;
    
    // 加载数据到 Shared Memory
    for (int d = threadIdx.z; d < depth; d += blockDim.z) {
        shared_data[d] = input[col_idx + d * height * width];
    }
    __syncthreads();
    
    // 执行 1D DCT
    for (int d = threadIdx.z; d < depth; d += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < depth; k++) {
            sum += shared_data[k] * dct_coeff[d][k];
        }
        output[col_idx + d * height * width] = sum;
    }
}

// 1D IDCT 变换（沿 Z 轴）
__global__ void idct1d_z_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    int c = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || h >= height || w >= width) return;
    
    __shared__ float shared_data[TILE_D];
    
    int col_idx = c * depth * height * width + h * width + w;
    
    for (int d = threadIdx.z; d < depth; d += blockDim.z) {
        shared_data[d] = input[col_idx + d * height * width];
    }
    __syncthreads();
    
    for (int d = threadIdx.z; d < depth; d += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < depth; k++) {
            sum += shared_data[k] * idct_coeff[d][k];
        }
        output[col_idx + d * height * width] = sum;
    }
}

// 1D DCT 变换（沿 Y 轴）
__global__ void dct1d_y_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    int c = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || d >= depth || w >= width) return;
    
    __shared__ float shared_data[TILE_H];
    
    int row_idx = c * depth * height * width + d * height * width + w;
    
    for (int h = threadIdx.z; h < height; h += blockDim.z) {
        shared_data[h] = input[row_idx + h * width];
    }
    __syncthreads();
    
    for (int h = threadIdx.z; h < height; h += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < height; k++) {
            sum += shared_data[k] * dct_coeff[h][k];
        }
        output[row_idx + h * width] = sum;
    }
}

// 1D IDCT 变换（沿 Y 轴）
__global__ void idct1d_y_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    int c = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || d >= depth || w >= width) return;
    
    __shared__ float shared_data[TILE_H];
    
    int row_idx = c * depth * height * width + d * height * width + w;
    
    for (int h = threadIdx.z; h < height; h += blockDim.z) {
        shared_data[h] = input[row_idx + h * width];
    }
    __syncthreads();
    
    for (int h = threadIdx.z; h < height; h += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < height; k++) {
            sum += shared_data[k] * idct_coeff[h][k];
        }
        output[row_idx + h * width] = sum;
    }
}

// 1D DCT 变换（沿 X 轴）
__global__ void dct1d_x_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    int c = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || d >= depth || h >= height) return;
    
    __shared__ float shared_data[TILE_W];
    
    int plane_idx = c * depth * height * width + d * height * width + h * width;
    
    for (int w = threadIdx.z; w < width; w += blockDim.z) {
        shared_data[w] = input[plane_idx + w];
    }
    __syncthreads();
    
    for (int w = threadIdx.z; w < width; w += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += shared_data[k] * dct_coeff[w][k];
        }
        output[plane_idx + w] = sum;
    }
}

// 1D IDCT 变换（沿 X 轴）
__global__ void idct1d_x_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int depth, int height, int width, int channels
) {
    int c = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels || d >= depth || h >= height) return;
    
    __shared__ float shared_data[TILE_W];
    
    int plane_idx = c * depth * height * width + d * height * width + h * width;
    
    for (int w = threadIdx.z; w < width; w += blockDim.z) {
        shared_data[w] = input[plane_idx + w];
    }
    __syncthreads();
    
    for (int w = threadIdx.z; w < width; w += blockDim.z) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += shared_data[k] * idct_coeff[w][k];
        }
        output[plane_idx + w] = sum;
    }
}

// C++ 接口函数
extern "C" {
    
// 初始化 DCT 系数
void init_dct_coefficients_cuda(int max_size) {
    dim3 grid(1, 1, 1);
    dim3 block(TILE_D, TILE_D, 1);
    init_dct_coefficients<<<grid, block>>>();
    cudaDeviceSynchronize();
}

// 3D DCT（分解为三次 1D DCT）
void dct3d_cuda(
    float* output,
    const float* input,
    int depth, int height, int width, int channels,
    cudaStream_t stream
) {
    // 沿 Z 轴进行 1D DCT
    dim3 block_z(TILE_W, TILE_H, TILE_D);
    dim3 grid_z((width + TILE_W - 1) / TILE_W,
                 (height + TILE_H - 1) / TILE_H,
                 (depth + TILE_D - 1) / TILE_D);
    
    float* temp_z;
    cudaMalloc(&temp_z, channels * depth * height * width * sizeof(float));
    
    dct1d_z_kernel<<<grid_z, block_z, 0, stream>>>(temp_z, input, depth, height, width, channels);
    
    // 沿 Y 轴进行 1D DCT
    dim3 block_y(TILE_W, TILE_D, TILE_H);
    dim3 grid_y((width + TILE_W - 1) / TILE_W,
                 (depth + TILE_D - 1) / TILE_D,
                 (height + TILE_H - 1) / TILE_H);
    
    float* temp_y;
    cudaMalloc(&temp_y, channels * depth * height * width * sizeof(float));
    
    dct1d_y_kernel<<<grid_y, block_y, 0, stream>>>(temp_y, temp_z, depth, height, width, channels);
    cudaFree(temp_z);
    
    // 沿 X 轴进行 1D DCT
    dim3 block_x(TILE_H, TILE_D, TILE_W);
    dim3 grid_x((height + TILE_H - 1) / TILE_H,
                 (depth + TILE_D - 1) / TILE_D,
                 (width + TILE_W - 1) / TILE_W);
    
    dct1d_x_kernel<<<grid_x, block_x, 0, stream>>>(output, temp_y, depth, height, width, channels);
    cudaFree(temp_y);
}

// 3D IDCT（分解为三次 1D IDCT）
void idct3d_cuda(
    float* output,
    const float* input,
    int depth, int height, int width, int channels,
    cudaStream_t stream
) {
    // 沿 X 轴进行 1D IDCT
    dim3 block_x(TILE_H, TILE_D, TILE_W);
    dim3 grid_x((height + TILE_H - 1) / TILE_H,
                 (depth + TILE_D - 1) / TILE_D,
                 (width + TILE_W - 1) / TILE_W);
    
    float* temp_x;
    cudaMalloc(&temp_x, channels * depth * height * width * sizeof(float));
    
    idct1d_x_kernel<<<grid_x, block_x, 0, stream>>>(temp_x, input, depth, height, width, channels);
    
    // 沿 Y 轴进行 1D IDCT
    dim3 block_y(TILE_W, TILE_D, TILE_H);
    dim3 grid_y((width + TILE_W - 1) / TILE_W,
                 (depth + TILE_D - 1) / TILE_D,
                 (height + TILE_H - 1) / TILE_H);
    
    float* temp_y;
    cudaMalloc(&temp_y, channels * depth * height * width * sizeof(float));
    
    idct1d_y_kernel<<<grid_y, block_y, 0, stream>>>(temp_y, temp_x, depth, height, width, channels);
    cudaFree(temp_x);
    
    // 沿 Z 轴进行 1D IDCT
    dim3 block_z(TILE_W, TILE_H, TILE_D);
    dim3 grid_z((width + TILE_W - 1) / TILE_W,
                 (height + TILE_H - 1) / TILE_H,
                 (depth + TILE_D - 1) / TILE_D);
    
    idct1d_z_kernel<<<grid_z, block_z, 0, stream>>>(output, temp_y, depth, height, width, channels);
    cudaFree(temp_y);
}
}
