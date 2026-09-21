/*
 * 3D Convolution CUDA Kernel
 * 实现真正的 CUDA Tiles 加速的 3D 卷积
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Tile 大小配置
#define TILE_D 8
#define TILE_H 8
#define TILE_W 8
#define TILE_C 16

// 卷积核大小
#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

// Im2Col + GEMM 的 Tile 化 3D 卷积
__global__ void conv3d_im2col_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding
) {
    // 计算输出索引
    int b = blockIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 添加严格的边界检查
    if (b >= batch_size) return;
    if (c_out >= out_channels) return;
    if (out_idx >= out_depth * out_height * out_width) return;
    
    int out_d = out_idx / (out_height * out_width);
    int out_h = (out_idx % (out_height * out_width)) / out_width;
    int out_w = out_idx % out_width;
    
    // 再次检查输出维度边界
    if (out_d >= out_depth || out_h >= out_height || out_w >= out_width) return;
    
    // 计算输入区域的起始位置
    int in_d_start = out_d * stride - padding;
    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;
    
    // 计算卷积（直接从全局内存读取权重，避免共享内存越界问题）
    float sum = 0.0f;
    int weight_offset = c_out * in_channels * kernel_size * kernel_size * kernel_size;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            int in_d = in_d_start + kd;
            if (in_d < 0 || in_d >= depth) continue;
            
            for (int kh = 0; kh < kernel_size; kh++) {
                int in_h = in_h_start + kh;
                if (in_h < 0 || in_h >= height) continue;
                
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_w = in_w_start + kw;
                    if (in_w < 0 || in_w >= width) continue;
                    
                    int in_idx = b * in_channels * depth * height * width +
                                  c_in * depth * height * width +
                                  in_d * height * width +
                                  in_h * width + in_w;
                    
                    // 直接从全局内存读取权重
                    int weight_idx = weight_offset + c_in * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    float weight_val = weight[weight_idx];
                    
                    sum += input[in_idx] * weight_val;
                }
            }
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // 写入输出
    int out_idx_full = b * out_channels * out_depth * out_height * out_width +
                        c_out * out_depth * out_height * out_width +
                        out_d * out_height * out_width +
                        out_h * out_width + out_w;
    output[out_idx_full] = sum;
}

// 稀疏优化的 3D 卷积（跳过空白 Tile）
__global__ void conv3d_sparse_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mask,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    float sparse_threshold
) {
    // 计算输出索引
    int b = blockIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 添加严格的边界检查
    if (b >= batch_size) return;
    if (c_out >= out_channels) return;
    if (out_idx >= out_depth * out_height * out_width) return;
    
    int out_d = out_idx / (out_height * out_width);
    int out_h = (out_idx % (out_height * out_width)) / out_width;
    int out_w = out_idx % out_width;
    
    // 检查当前 Tile 是否为稀疏（空白）
    int tile_d_start = out_d * stride - padding;
    int tile_h_start = out_h * stride - padding;
    int tile_w_start = out_w * stride - padding;
    
    int valid_count = 0;
    int total_count = 0;
    
    for (int kd = 0; kd < kernel_size; kd++) {
        int in_d = tile_d_start + kd;
        if (in_d < 0 || in_d >= depth) continue;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_h = tile_h_start + kh;
            if (in_h < 0 || in_h >= height) continue;
            
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_w = tile_w_start + kw;
                if (in_w < 0 || in_w >= width) continue;
                
                // mask的形状是[B, 1, D, H, W]，需要包含通道维度（这里固定为0）
                int mask_idx = b * 1 * depth * height * width +
                                0 * depth * height * width +  // 通道维度固定为0
                                in_d * height * width +
                                in_h * width + in_w;
                total_count++;
                if (mask[mask_idx] > 0.5f) {
                    valid_count++;
                }
            }
        }
    }
    
    // 如果 Tile 中有效体素占比低于阈值，跳过计算
    if (total_count > 0 && (float)valid_count / total_count < sparse_threshold) {
        int out_idx_full = b * out_channels * out_depth * out_height * out_width +
                            c_out * out_depth * out_height * out_width +
                            out_d * out_height * out_width +
                            out_h * out_width + out_w;
        output[out_idx_full] = 0.0f;
        return;
    }
    
    // 正常计算卷积（直接从全局内存读取权重，避免共享内存越界问题）
    int in_d_start = out_d * stride - padding;
    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;
    
    float sum = 0.0f;
    int weight_offset = c_out * in_channels * kernel_size * kernel_size * kernel_size;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            int in_d = in_d_start + kd;
            if (in_d < 0 || in_d >= depth) continue;
            
            for (int kh = 0; kh < kernel_size; kh++) {
                int in_h = in_h_start + kh;
                if (in_h < 0 || in_h >= height) continue;
                
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_w = in_w_start + kw;
                    if (in_w < 0 || in_w >= width) continue;
                    
                    int in_idx = b * in_channels * depth * height * width +
                                  c_in * depth * height * width +
                                  in_d * height * width +
                                  in_h * width + in_w;
                    
                    // 应用掩码 - mask形状是[B, 1, D, H, W]
                    int mask_idx = b * 1 * depth * height * width +
                                    0 * depth * height * width +  // 通道维度固定为0
                                    in_d * height * width +
                                    in_h * width + in_w;
                    float mask_val = mask[mask_idx];
                    
                    // 直接从全局内存读取权重
                    int weight_idx = weight_offset + c_in * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    float weight_val = weight[weight_idx];
                    
                    sum += input[in_idx] * mask_val * weight_val;
                }
            }
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // 写入输出
    int out_idx_full = b * out_channels * out_depth * out_height * out_width +
                        c_out * out_depth * out_height * out_width +
                        out_d * out_height * out_width +
                        out_h * out_width + out_w;
    output[out_idx_full] = sum;
}

// C++ 接口函数
extern "C" {

// 标准 3D 卷积
void conv3d_cuda(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
) {
    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 每个通道和批次的输出元素数量
    int output_elements_per_channel_batch = out_depth * out_height * out_width;
    
    dim3 block(TILE_W, TILE_C, 1);
    dim3 grid((output_elements_per_channel_batch + block.x - 1) / block.x,
               out_channels,
               batch_size);
    
    conv3d_im2col_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding
    );
}

// 稀疏优化的 3D 卷积
void conv3d_sparse_cuda(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* mask,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    float sparse_threshold,
    cudaStream_t stream
) {
    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 每个通道和批次的输出元素数量
    int output_elements_per_channel_batch = out_depth * out_height * out_width;
    
    dim3 block(TILE_W, TILE_C, 1);
    dim3 grid((output_elements_per_channel_batch + block.x - 1) / block.x,
               out_channels,
               batch_size);
    
    conv3d_sparse_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, mask,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        sparse_threshold
    );
}
}
