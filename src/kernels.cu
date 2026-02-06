
// #include <__clang_cuda_math.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "./block_attention.cuh"
#include "./log.h"
#include "./utils.cuh"
template <typename T>
__inline__ __device__ T warpReduceSum(T val);
template <typename T>
__global__ void traceKernel(const T* mat_input, T* output,

                            const size_t rows, const size_t cols, const size_t ndiag);
/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t n_diag = std::min(rows, cols);
  ASSERT_MSG(h_input.size() == rows * cols, "kernels::trace: wrong h_input size");
  if (h_input.empty() || rows == 0 || cols == 0) return T(0);
  size_t bytes_matrix = rows * cols * sizeof(T);
  T *d_input, *d_output;
  CHECK_CUDA(cudaMalloc((void**)&d_input, bytes_matrix));
  CHECK_CUDA(cudaMalloc((void**)&d_output, sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes_matrix, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_output, 0, sizeof(T)));  // 初始化为 0
  // 配置 Kernel 参数
  int blockSize = 256;
  // 根据对角线元素数量计算需要的 Block 数量，但要设置上限避免过多小任务
  int numBlocks = (n_diag + blockSize - 1) / blockSize;
  // 限制 grid 大小，利用 Grid-Stride Loop
  numBlocks = std::min(numBlocks, 1024);
  traceKernel<T><<<numBlocks, blockSize>>>(d_input, d_output, rows, cols, n_diag);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 拷贝结果回 Host
  T result;
  CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost));

  //  释放内存
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));

  return result;
}
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  // 假设 warpSize 为 32
  // 使用 __shfl_down_sync 进行蝴蝶形交换求和
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}
template <typename T>
__global__ void traceKernel(const T* mat_input, T* output,

                            const size_t rows, const size_t cols, const size_t ndiag) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;  // 即线程总数
  T local_sum = static_cast<T>(0);
  // 使用 grid-stride loop 遍历对角线索引
  // 对角线元素的矩阵索引 = diag_idx * (cols + 1)
  for (size_t diag_idx = tid; diag_idx < ndiag; diag_idx += stride) {
    size_t matrix_idx = diag_idx * (cols + 1);  // 计算对角线元素在矩阵中的索引
    local_sum += mat_input[matrix_idx];
  }
  // warp级规约，不需要同步，因为同一个warp是同时完成的
  local_sum = warpReduceSum(local_sum);
  // block级别规约
  // 收集block内所有warp的数据到shared_sums，1024(block最大线程数)/32(warp里线程数)=32(warp数量)
  static __shared__ T shared_sums[32];
  size_t lane = threadIdx.x % warpSize;
  size_t warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared_sums[warp_id] = local_sum;
  }
  __syncthreads();
  // 让第一个 Warp 规约smem中的结果
  local_sum = (threadIdx.x < blockDim.x / warpSize) ? shared_sums[lane] : 0;

  if (warp_id == 0) {
    local_sum = warpReduceSum(local_sum);
  }

  // Global Reduction: Block 的结果写入全局内存
  // 由 Block 的第 0 号线程执行原子加
  if (threadIdx.x == 0) {
    atomicAdd(output, local_sum);
  }
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
/* # 输入
Q: [batch_size, tgt_seq_len, query_heads, head_dim]
K: [batch_size, src_seq_len, kv_heads, head_dim]
V: [batch_size, src_seq_len, kv_heads, head_dim]

# 对每个 (batch, head) 组合分别计算
for b in range(batch_size):
   for h in range(query_heads):
       # 提取2D矩阵
       Q_2d = Q[b, :, h, :]      # [tgt_seq_len, head_dim]
       K_2d = K[b, :, kv_head, :] # [src_seq_len, head_dim]
       V_2d = V[b, :, kv_head, :] # [src_seq_len, head_dim]

       # 二维矩阵计算
       scores = Q_2d @ K_2d.T     # [tgt_seq_len, src_seq_len]
       scores = softmax(scores)   # [tgt_seq_len, src_seq_len]
       output = scores @ V_2d     # [tgt_seq_len, head_dim]

       # 存储结果
       O[b, :, h, :] = output     # [tgt_seq_len, head_dim] */
template <typename T>
std::vector<size_t> compute_Br_Bc(size_t max_shm_per_block, int head_dim);
int get_sram_size(int device_id = 0);
template <typename T>
__global__ void kernel_flash_attn_v2(T* d_q, T* d_k, T* d_v, T* d_o, int batch_size, int target_seq_len,
                                     int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal,
                                     int Br, int Bc, int Tr, int Tc, int* q_strides, int* k_strides, int* v_strides,
                                     int* o_strides);
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k, const std::vector<T>& h_v,
                    std::vector<T>& h_o, int batch_size, int target_seq_len, int src_seq_len, int query_heads,
                    int kv_heads, int head_dim, bool is_causal) {
  // 一些日志
  LOG_INFO("batch_size=%d", batch_size);
  LOG_INFO("target_seq_len=%d", target_seq_len);
  LOG_INFO("src_seq_len=%d", src_seq_len);
  LOG_INFO("query_heads=%d", query_heads);
  LOG_INFO("kv_heads=%d", kv_heads);
  LOG_INFO("head_dim=%d", head_dim);
  // TODO: Implement the flash attention function
  // 计算并行的模式需要的参数，当前的实现不分batch
  // Br,Bc分别为q和k,v的分块行数大小
  size_t max_shm_per_block = get_sram_size(0);
  /*   auto B_size = compute_Br_Bc<T>(max_shm_per_block, head_dim);
    size_t Br = B_size[0];
    size_t Bc = B_size[1]; */
  // 先固定一下 Br Bc看看
  size_t Br = 32; 
  size_t Bc = 32;
  // 设置kernel launch参数
  int Tr = (target_seq_len + Br - 1) / Br;
  int Tc = (src_seq_len + Bc - 1) / Bc;
  dim3 gridDim;
  gridDim.x = batch_size;   // 每个 batch 一个维度
  gridDim.y = query_heads;  // 每个 query head 一个维度
  gridDim.z = Tr;           // 每个 Q tile 行块一个维度
  int blockSize = 128;
  dim3 blockDim;
  blockDim.x = blockSize;
  blockDim.y = 1;
  blockDim.z = 1;
  // 计算 q, k, v 的 strides
  // Q 形状: [batch_size, tgt_seq_len, query_heads, head_dim]
  // K, V 形状: [batch_size, src_seq_len, kv_heads, head_dim]
  std::vector<int> q_strides(4, 1);
  std::vector<int> k_strides(4, 1);
  std::vector<int> v_strides(4, 1);
  std::vector<int> o_strides(4, 1);
  // Q 的 strides（从最内层到最外层）
  q_strides[3] = 1;                              // head_dim 维度
  q_strides[2] = q_strides[3] * head_dim;        // query_heads 维度
  q_strides[1] = q_strides[2] * query_heads;     // tgt_seq_len 维度
  q_strides[0] = q_strides[1] * target_seq_len;  // batch_size 维度

  // K 和 V 的 strides（形状相同）
  k_strides[3] = 1;                           // head_dim 维度
  k_strides[2] = k_strides[3] * head_dim;     // kv_heads 维度
  k_strides[1] = k_strides[2] * kv_heads;     // src_seq_len 维度
  k_strides[0] = k_strides[1] * src_seq_len;  // batch_size 维度

  v_strides[3] = 1;                           // head_dim 维度
  v_strides[2] = v_strides[3] * head_dim;     // kv_heads 维度
  v_strides[1] = v_strides[2] * kv_heads;     // src_seq_len 维度
  v_strides[0] = v_strides[1] * src_seq_len;  // batch_size 维度

  o_strides[3] = 1;                              // head_dim 维度
  o_strides[2] = o_strides[3] * head_dim;        // kv_heads 维度
  o_strides[1] = o_strides[2] * query_heads;     // src_seq_len 维度
  o_strides[0] = o_strides[1] * target_seq_len;  // batch_size 维度
  // 设置launch kernel的sram_size，注意对齐访问
  size_t t_bytes = (2 * (Bc + Br) * head_dim) * sizeof(T);
  size_t align_t = alignof(float);
  size_t t_bytes_aligned = (t_bytes + align_t - 1) / align_t * align_t;
  using AccumT = std::conditional_t<std::is_same_v<T, float>, double, float>;
  size_t s_bytes = Bc * Br * sizeof(AccumT);

  size_t lm_bytes = 2 * Br * sizeof(AccumT);
  size_t s_bytes_aligned = (t_bytes_aligned + s_bytes + alignof(AccumT) - 1) / alignof(AccumT) * alignof(AccumT);
  size_t sram_size = s_bytes_aligned + lm_bytes;
  // 数据搬运到global memory上
  // 创建device侧指针
  T* d_q = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_q, h_q.size() * sizeof(T)));
  T* d_k = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_k, h_k.size() * sizeof(T)));
  T* d_v = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_v, h_v.size() * sizeof(T)));
  T* d_o = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_o, h_o.size() * sizeof(T)));
  int* d_q_strides = nullptr;
  int* d_k_strides = nullptr;
  int* d_v_strides = nullptr;
  int* d_o_strides = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_q_strides, q_strides.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_k_strides, k_strides.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_v_strides, v_strides.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void**)&d_o_strides, o_strides.size() * sizeof(int)));
  // 数据搬运
  CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_o, 0, h_o.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_q_strides, q_strides.data(), q_strides.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_strides, k_strides.data(), k_strides.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v_strides, v_strides.data(), v_strides.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_o_strides, o_strides.data(), o_strides.size() * sizeof(int), cudaMemcpyHostToDevice));
  // 启动kernel函数
  kernel_flash_attn_v2<T><<<gridDim, blockDim, sram_size>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal,
      static_cast<int>(Br), static_cast<int>(Bc), Tr, Tc, d_q_strides, d_k_strides, d_v_strides, d_o_strides);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  // 拷回结果
  CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost));
  // 释放显存
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_o));
  CHECK_CUDA(cudaFree(d_q_strides));
  CHECK_CUDA(cudaFree(d_k_strides));
  CHECK_CUDA(cudaFree(d_v_strides));
  CHECK_CUDA(cudaFree(d_o_strides));
}
// 不清楚head_dim是多少，sram不能编译期计算出大小，所以选择使用偏移量方式来加载不同矩阵
template <typename T>
__global__ void kernel_flash_attn_v2(T* d_q, T* d_k, T* d_v, T* d_o, int batch_size, int target_seq_len,
                                     int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal,
                                     int Br, int Bc, int Tr, int Tc, int* q_strides, int* k_strides, int* v_strides,
                                     int* o_strides) {
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int q_chunk_idx = blockIdx.z;                // 当前block负责哪些Qi块
  extern __shared__ unsigned char sram_raw[];  // sram_size这么大的sram
  // 通过偏移量来分割不同矩阵
  char* sram_bytes = reinterpret_cast<char*>(sram_raw);
  T* q_s = reinterpret_cast<T*>(sram_bytes);  // Q tile: head_dim * Br 个元素
  T* k_s = q_s + head_dim * Br;               // K tile: head_dim * Bc 个元素
  T* v_s = k_s + head_dim * Bc;               // V tile: head_dim * Bc 个元素
  T* o_s = v_s + head_dim * Bc;               // O tile: head_dim * Br 个元素
  using AccumT = std::conditional_t<std::is_same_v<T, float>, double, float>;
  size_t t_bytes = (2 * (Bc + Br) * head_dim) * sizeof(T);
  size_t align_t = alignof(float);
  size_t t_bytes_aligned = (t_bytes + align_t - 1) / align_t * align_t;
  AccumT* s_s = reinterpret_cast<AccumT*>(sram_bytes + t_bytes_aligned);  // S=QK^T: Br*Bc个元素
  size_t s_bytes = Bc * Br * sizeof(AccumT);
  size_t s_bytes_aligned = (t_bytes_aligned + s_bytes + alignof(AccumT) - 1) / alignof(AccumT) * alignof(AccumT);
  AccumT* l_s = reinterpret_cast<AccumT*>(sram_bytes + s_bytes_aligned);  // 分母累加器:Br个元素
  AccumT* m_s = l_s + Br;                                                 // 最大值记录累加器:Br个元素
  // step1:从hbm加载Qtile
  // 计算每个block拿到的矩阵
  // Q 形状: [batch_size, tgt_seq_len, query_heads, head_dim]
  // 当前 block 处理: batch=batch_idx, head=head_idx, 行范围=[q_chunk_idx*Br, (q_chunk_idx+1)*Br)
  long long d_q_offset = (long long)batch_idx * q_strides[0] +           // batch 维度偏移
                         (long long)(q_chunk_idx * Br) * q_strides[1] +  // tgt_seq_len 维度偏移（起始行）
                         (long long)head_idx * q_strides[2];             // query_heads 维度偏移
                                                                         // 加载矩阵
  int q_valid_rows = max(0, target_seq_len - q_chunk_idx * Br);
  q_valid_rows = min(q_valid_rows, Br);
  sync_load_gmem_smemV1_masked(d_q + d_q_offset, q_s, Br, head_dim, q_strides[1], q_valid_rows);
  // step2:初始化 l,m,o
  sync_init_sram_V1<AccumT>(l_s, static_cast<AccumT>(0), Br);
  sync_init_sram_V1<AccumT>(m_s, static_cast<AccumT>(-CUDART_INF_F), Br);
  sync_init_sram_V1<T>(o_s, static_cast<T>(0), Br * head_dim);
  // step3:循环加载 Kj Vj
  // K的shape:[batch_size, src_seq_len, kv_heads, head_dim]
  // V的shape:[batch_size, src_seq_len, kv_heads, head_dim]
  for (size_t j = 0; j < Tc; j++) {
    // 确定 KV的head_idx，假如要做GQA计算的话，那么Q头会更多，需要确定当前的Q头应该对应哪个KV头
    int group_size = query_heads / kv_heads;
    int kv_head_idx = head_idx / group_size;
    long long d_k_offset = (long long)batch_idx * k_strides[0] + (long long)(j * Bc) * k_strides[1] +
                           (long long)kv_head_idx * k_strides[2];
    long long d_v_offset = (long long)batch_idx * v_strides[0] + (long long)(j * Bc) * v_strides[1] +
                           (long long)kv_head_idx * v_strides[2];
    int k_valid_rows = max(0, src_seq_len - static_cast<int>(j) * Bc);
    k_valid_rows = min(k_valid_rows, Bc);
    sync_load_gmem_smemV1_masked(d_k + d_k_offset, k_s, Bc, head_dim, k_strides[1], k_valid_rows);
    sync_load_gmem_smemV1_masked(d_v + d_v_offset, v_s, Bc, head_dim, v_strides[1], k_valid_rows);
    // 计算片上的 S=softmax(QK^T/scale+Mask)V,并更新m和l
    // 计算scale,q_row_start,k_col_start
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    int q_row_start = q_chunk_idx * Br;
    int k_col_start = j * Bc;
    block_attentionV1<T, AccumT>(q_s, k_s, v_s, s_s, o_s, l_s, m_s, is_causal, Br, Bc, head_dim, head_dim, head_dim,
                                 head_dim, Bc, head_dim, q_row_start, k_col_start, target_seq_len, src_seq_len, scale);
  }
  __syncthreads();
  // 延迟归一化：O_final = O_unnorm / l
  for (int idx = threadIdx.x; idx < q_valid_rows * head_dim; idx += blockDim.x) {
    int row = idx / head_dim;
    AccumT l_val = l_s[row];
    if (l_val > static_cast<AccumT>(0)) {
      if constexpr (std::is_same_v<T, half>) {
        float o_val = __half2float(o_s[idx]);
        o_s[idx] = __float2half_rn(o_val / static_cast<float>(l_val));
      } else {
        float o_val = static_cast<float>(o_s[idx]);
        if constexpr (std::is_same_v<AccumT, double>) {
          o_s[idx] = static_cast<T>(static_cast<double>(o_val) / l_val);
        } else {
          o_s[idx] = static_cast<T>(o_val / static_cast<float>(l_val));
        }
      }
    }
  }
  __syncthreads();
  // o_s写回d_o
  // d_o 的shape[batch_size, tgt_seq_len, query_heads, head_dim]
  // step1:先确定当前block需要写回d_o的offset
  long long d_o_offset = (long long)batch_idx * o_strides[0] + (long long)(q_chunk_idx * Br) * o_strides[1] +
                         (long long)head_idx * o_strides[2];
  // step2:o_s写回d_o（处理尾块行）
  int q_valid_rows_o = max(0, target_seq_len - q_chunk_idx * Br);
  q_valid_rows_o = min(q_valid_rows_o, Br);
  int elements = q_valid_rows_o * head_dim;
  for (int idx = threadIdx.x; idx < elements; idx += blockDim.x) {
    int row = idx / head_dim;
    int col = idx % head_dim;
    d_o[d_o_offset + (long long)row * o_strides[1] + col] = o_s[row * head_dim + col];
  }
}
// 计算需要分配的Br,Bc大小
//
template <typename T>
std::vector<size_t> compute_Br_Bc(size_t max_shm_per_block, int head_dim) {
  size_t row_size = head_dim * sizeof(T) * 2;  // Br和Bc每一行所需要的内存
  // 百分之40留给L1 cache
  size_t Br = (row_size / 5) / sizeof(T);
  size_t Bc = Br * 2;
  return {Br, Bc};
}
// 动态获取每个block能申请到的sram大小
int get_sram_size(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  // 关键字段：sharedMemPerMultiprocessor
  // 这代表物理上每个 SM (Streaming Multiprocessor) 拥有的 SRAM 总量
  size_t sram_per_sm = prop.sharedMemPerMultiprocessor;

  size_t max_shm_per_block = prop.sharedMemPerBlockOptin;

  return max_shm_per_block;
}
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                                    std::vector<float>&, int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&, const std::vector<half>&,
                                   std::vector<half>&, int, int, int, int, int, int, bool);
