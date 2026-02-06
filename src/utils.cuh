// 实现一些功能性kernel函数
#include <cuda_fp16.h>
#include <cuda_runtime.h>
/**
 * @brief 同步从gmem加载到smem的函数，模板实现v1
 * * @tparam T 数据类型 (float, half, __nv_bfloat16 等)
 * @param g_ptr   全局内存起始地址 (已经加上了 batch 和 head 的 offset)
 * @param s_ptr   共享内存起始地址
 * @param Br      分块的行数 (Block Rows)
 * @param D       分块的列数 (Head Dim)
 * @param g_stride Global Memory 中每一行的步长 (stride_seq)
 */
template <typename T>
__device__ void sync_load_gmem_smemV1(const T* __restrict__ g_ptr, T* __restrict__ s_ptr, const int Br, const int D,
                                      const int g_stride);
/**
 * @brief 同步从gmem加载到smem，支持尾块行填0
 * @tparam T 数据类型
 * @param g_ptr   全局内存起始地址（已加上 tile 起始偏移）
 * @param s_ptr   共享内存起始地址
 * @param Br      分块行数
 * @param D       分块列数
 * @param g_stride Global Memory 中每一行步长
 * @param valid_rows 本 tile 内有效行数（<Br 的部分会填0）
 */
template <typename T>
__device__ void sync_load_gmem_smemV1_masked(const T* __restrict__ g_ptr, T* __restrict__ s_ptr, const int Br,
                                             const int D, const int g_stride, const int valid_rows);
/**
 * @brief 初始化一块sram内存的函数
 * * @tparam T 数据类型 (float, half, __nv_bfloat16 等)
 * @param s_ptr   共享内存起始地址
 * @param val      需要初始化的值
 * @param n        初始化元素个数
 */
template <typename T>
__device__ void sync_init_sram_V1(T* __restrict__ s_ptr, const T val, const int n);