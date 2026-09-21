/*
 * PyBind C++ 接口（PyTorch 2.4.0 兼容）
 * 连接 CUDA kernel 和 Python
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel 声明
extern "C" {
    void init_dct_coefficients_cuda(int max_size);
    void dct3d_cuda(float* output, const float* input,
                     int depth, int height, int width, int channels,
                     cudaStream_t stream);
    void idct3d_cuda(float* output, const float* input,
                      int depth, int height, int width, int channels,
                      cudaStream_t stream);
    void conv3d_cuda(float* output, const float* input,
                      const float* weight, const float* bias,
                      int batch_size, int in_channels, int out_channels,
                      int depth, int height, int width,
                      int kernel_size, int stride, int padding,
                      cudaStream_t stream);
    void conv3d_sparse_cuda(float* output, const float* input,
                            const float* weight, const float* bias,
                            const float* mask,
                            int batch_size, int in_channels, int out_channels,
                            int depth, int height, int width,
                            int kernel_size, int stride, int padding,
                            float sparse_threshold, cudaStream_t stream);
}

// 3D DCT 算子
torch::Tensor dct3d_cuda_forward(
    torch::Tensor input,
    int max_tile_size
) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)");
    
    // 获取输入形状
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    // 创建输出张量
    auto output = torch::empty_like(input);
    
    // 初始化 DCT 系数
    init_dct_coefficients_cuda(max_tile_size);
    
    // 执行 3D DCT
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    dct3d_cuda(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        depth, height, width, channels,
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output;
}

// 3D IDCT 算子
torch::Tensor idct3d_cuda_forward(
    torch::Tensor input,
    int max_tile_size
) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)");
    
    // 获取输入形状
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    // 创建输出张量
    auto output = torch::empty_like(input);
    
    // 执行 3D IDCT
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    idct3d_cuda(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        depth, height, width, channels,
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output;
}

// 3D 卷积算子
torch::Tensor conv3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int kernel_size,
    int stride,
    int padding
) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D tensor (O, I, K, K, K)");
    
    // 获取输入形状
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    // 计算输出形状
    auto out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    auto out_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 创建输出张量
    auto output = torch::zeros(
        {batch_size, out_channels, out_depth, out_height, out_width},
        input.options()
    );
    
    // 执行 3D 卷积
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 获取 bias 指针（如果存在）
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;
    
    conv3d_cuda(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output;
}

// 稀疏优化的 3D 卷积算子
torch::Tensor conv3d_sparse_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    torch::Tensor mask,
    int kernel_size,
    int stride,
    int padding,
    float sparse_threshold
) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "Mask must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)");
    TORCH_CHECK(mask.dim() == 5, "Mask must be 5D tensor (B, 1, D, H, W)");
    
    // 确保 mask 的通道数为 1
    TORCH_CHECK(mask.size(1) == 1, "Mask must have 1 channel");
    
    // 获取输入形状
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    // 计算输出形状
    auto out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    auto out_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    // 创建输出张量
    auto output = torch::zeros(
        {batch_size, out_channels, out_depth, out_height, out_width},
        input.options()
    );
    
    // 执行稀疏优化的 3D 卷积
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 获取 bias 指针（如果存在）
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;
    
    conv3d_sparse_cuda(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        mask.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        sparse_threshold,
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output;
}

// PyBind 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "3D-vHeat CUDA Tiles 加速算子";
    
    // 3D DCT/IDCT
    m.def("dct3d_forward", &dct3d_cuda_forward,
          "3D DCT 前向传播",
          py::arg("input"),
          py::arg("max_tile_size") = 32);
    
    m.def("idct3d_forward", &idct3d_cuda_forward,
          "3D IDCT 前向传播",
          py::arg("input"),
          py::arg("max_tile_size") = 32);
    
    // 3D 卷积
    m.def("conv3d_forward", &conv3d_cuda_forward,
          "3D 卷积前向传播",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("kernel_size") = 3,
          py::arg("stride") = 1,
          py::arg("padding") = 0);
    
    m.def("conv3d_sparse_forward", &conv3d_sparse_cuda_forward,
          "稀疏优化的 3D 卷积前向传播",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("mask"),
          py::arg("kernel_size") = 3,
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("sparse_threshold") = 0.9);
}
