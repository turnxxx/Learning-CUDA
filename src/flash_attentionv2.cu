
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <type_traits>

#include "kernels.cu"
#include "utils.cuh"
/* 改进点
1. 中间过程S,O,m,l全用寄存器变量维护
2. HBM->SRAM的异步拷贝机制
3. wmma指令计算 gemm
4. 编译期优化
Q:[batch_size, tgt_seq_len, query_heads, head_dim]
K:[batch_size, src_seq_len, kv_heads, head_dim]
V:[batch_size, src_seq_len, kv_heads, head_dim]
*/
template <typename T, int Br, int Bc, int Rr, int Rc>
__global__ void kernel_flash_attn_v2(T* d_q, T* d_k, T* d_v, T* d_o, int batch_size, int target_seq_len,
                                     int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal,
                                     int Tc, int Tr, int* q_strides, int* k_strides, int* v_strides, int* o_strides) {
  // 计算当前blockThread需要处理的块
  constexpr int PAD = 16 / sizeof(T);
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int q_chunk_idx = blockIdx.z;
  // 设置内存
  int stride_s = head_dim + PAD;
  extern __shared__ unsigned char sram_raw[];
  T* q_s = reinterpret_cast<T*>(sram_raw);  // Q的[Br,head_dim]块
  T* k_s = q_s + Br * stride_s;             // K的[Bc,head_dim+PAD]块
  T* v_s = k_s + Bc * stride_s;             // V的[Bc,head_dim+PAD]块
  // 寄存器上的中间变量
  // 假设Tr=Tc=16(适配wmma指令)，Br=Bc=32，那么一个线程需要用到的寄存器是
  // T=float:320个(blockSize<204)，T=half/bf16，192个(blockSize<341)
  T p_r[Rr][Rc];  // 寄存器上的小块
  float l_r[Br];  // 寄存器上的累加器
  float m_r[Br];  // 寄存器上的最大值累加器
// 初始化寄存器数组
#pragma unroll
  for (int i = 0; i < Tr; ++i) {
#pragma unroll
    for (int j = 0; j < Tc; ++j) {
      p_r[i][j] = 0.0f;
    }
  }

#pragma unroll
  for (int i = 0; i < Br; ++i) {
    l_r[i] = 0.0f;
    m_r[i] = -INFINITY;  // Max 初始值通常设为负无穷
  }

  // 将Q拷贝到sram，因为Q是整个计算流程中不变的, 所以采用同步拷贝
  long long d_q_offset = (long long)batch_idx * q_strides[0] +           // batch 维度偏移
                         (long long)(q_chunk_idx * Br) * q_strides[1] +  // tgt_seq_len 维度偏移（起始行）
                         (long long)head_idx * q_strides[2];
  int q_valid_rows = max(0, target_seq_len - q_chunk_idx * Br);
  // 当前block下，实际需要拷贝的q的行数
  q_valid_rows = min(q_valid_rows, Br);
  sync_load_gmem_smemV1_masked(d_q + d_q_offset, q_s, Br, head_dim, q_strides[1], q_valid_rows);
  // 流水线启动，需要head_dim是4的倍数(float) 或者8的倍数(bf16,half)
  // 因为异步拷贝有16字节内存对齐的要求
  // __pipeline_memcpy__async智能拷贝 4 8
  // 16字节，所以需要一整个block内的线程并行执行 计算不同线程的拷贝的地址
  // 计算当前block拷贝的KV-offset
  int group_size = query_heads / kv_heads;
  int kv_head_idx = head_idx / group_size;
  long long d_k_offset = (long long)batch_idx * k_strides[0] + (long long)kv_head_idx * k_strides[2];
  long long d_v_offset = (long long)batch_idx * v_strides[0] + (long long)kv_head_idx * v_strides[2];
  // 计算第一轮处理时每个block处理K V的实际行数
  int k_valid_rows = max(src_seq_len, Bc);
  int v_valid_rows = max(src_seq_len, Bc);
  // 检查head_dim是否是16的整数倍
  assert(head_dim % 16 == 0 && "kernel_flash_attn_v2: Invalid head_dim, head_dim must be a multiple of 16");
  // 计算需要拷贝的总元素个数 (假设 Bc 和 head_dim 已经定义)
  int total_elements = Bc * head_dim;

  // 异步拷贝每次拷贝 16 字节
  constexpr int copy_bytes = 16;
  constexpr int elems_per_copy = copy_bytes / sizeof(T);  // float: 4, half/bf16: 8

  // 计算总共需要多少个 "copy_bytes 包"
  int total_packs = total_elements / elems_per_copy;
  // 异步拷贝一次加载4字节的话，只用padding无法避免 bank conflict,需要xor swizzling以及引入cute
  // 每个线程负责搬运多个包，步长为 blockDim.x

  for (int i = threadIdx.x; i < total_packs; i += blockDim.x) {
    // 计算当前包在 Tile 中的起始元素索引
    int element_idx = i * elems_per_copy;

    // 1. 计算源地址 (Global Memory)
    int row = element_idx / head_dim;
    int col = element_idx % head_dim;

    // 计算 Global Memory 中的实际偏移 (考虑 k_strides)
    // 注意：d_k_offset 已经是基地址偏移了
    // 假设 k_strides[1] 是行 stride (src_seq_len * head_dim 或其他)
    // 但在这里通常我们只处理一个 Tile，Tile 内通常是连续的或者 stride = head_dim
    // 如果是标准的 [B, N, H, D] 布局，head_dim 维度是连续的。
    // src_ptr = d_k + d_k_offset + row * k_strides[1] + col;
    // 简化版（假设 Tile 内连续，或者 stride = head_dim）：
    const T* src_ptr_k = d_k + d_k_offset + element_idx;

    // 2. 计算目标地址 (Shared Memory)
    T* dst_ptr_k = k_s + row * stride_s + col;  // q_k 是 K 在 Shared Memory 的起始地址

    // 3. 执行异步拷贝
    __pipeline_memcpy_async(dst_ptr_k, src_ptr_k, copy_bytes);
  }
  __pipeline_commit();
  for (int i = threadIdx.x; i < total_packs; i += blockDim.x) {
    int element_idx = i * elems_per_copy;

    // 1. 计算源地址 (Global Memory)
    int row = element_idx / head_dim;
    int col = element_idx % head_dim;

    const T* src_ptr_v = d_v + d_v_offset + element_idx;

    // 2. 计算目标地址 (Shared Memory)
    T* dst_ptr_v = v_s + row * stride_s + col;  // q_v 是 V 在 Shared Memory 的起始地址

    // 3. 执行异步拷贝
    __pipeline_memcpy_async(dst_ptr_v, src_ptr_v, copy_bytes);
  }
  __pipeline_commit();
  // 计算scale
  float scale = rsqrtf(head_dim);
  // group 0->K的拷贝，group 1 -> V的拷贝
  // main loop:遍历Tc
  for (int j = 0; j < Tc; j++) {
    // 等待K拷贝完成
    __pipeline_wait_prior(1);
    __syncthreads();
    int q_row_start = q_chunk_idx * Br;
    int k_col_start = j * Bc;
    // 接下来计算tile级别的softmax(QK^T+M),要使用wmma指令计算
  }
}