// #include <__clang_cuda_builtin_vars.h>

#include "./utils.cuh"
// 同步从gmem加载到smem的功能
template <typename T>
__device__ void sync_load_gmem_smemV1(const T* __restrict__ g_ptr, T* __restrict__ s_ptr, const int Br, const int D,
                                      const int g_stride) {
  // 需要加载元素总数
  size_t element_number = Br * D;
  for (size_t i = threadIdx.x; i < element_number; i += blockDim.x) {
    size_t row = i / D;
    size_t col = i % D;  // 一般来说D是2的幂次，可以用t &(D-1)代替
    size_t offset = row * g_stride + col;
    s_ptr[i] = g_ptr[offset];
  }
  __syncthreads();
}
template <typename T>
__device__ void sync_load_gmem_smemV1_masked(const T* __restrict__ g_ptr, T* __restrict__ s_ptr, const int Br,
                                             const int D, const int g_stride, const int valid_rows) {
  // 需要加载元素总数
  size_t element_number = Br * D;
  for (size_t i = threadIdx.x; i < element_number; i += blockDim.x) {
    size_t row = i / D;
    size_t col = i % D;
    if (row < static_cast<size_t>(valid_rows)) {
      size_t offset = row * g_stride + col;
      s_ptr[i] = g_ptr[offset];
    } else {
      s_ptr[i] = static_cast<T>(0);
    }
  }
  __syncthreads();
}
template <typename T>
__device__ void sync_init_sram_V1(T* __restrict__ s_ptr, const T val, const int n) {
  // 并行实现
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    s_ptr[i] = val;
  }
  __syncthreads();
}

// Explicit template instantiations for device functions (needed for separate compilation)
template __device__ void sync_load_gmem_smemV1<float>(const float*, float*, const int, const int, const int);
template __device__ void sync_load_gmem_smemV1<half>(const half*, half*, const int, const int, const int);
template __device__ void sync_load_gmem_smemV1_masked<float>(const float*, float*, const int, const int, const int,
                                                             const int);
template __device__ void sync_load_gmem_smemV1_masked<half>(const half*, half*, const int, const int, const int,
                                                            const int);
template __device__ void sync_init_sram_V1<float>(float*, const float, const int);
template __device__ void sync_init_sram_V1<half>(half*, const half, const int);
template __device__ void sync_init_sram_V1<double>(double*, const double, const int);