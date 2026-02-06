#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "./log.h"

// Shuffle helpers for double precision
template <typename T>
__device__ inline T warp_shuffle_down(T val, int offset) {
  return __shfl_down_sync(0xffffffff, val, offset);
}

template <>
__device__ inline double warp_shuffle_down(double val, int offset) {
  int2 a = *reinterpret_cast<int2*>(&val);
  a.x = __shfl_down_sync(0xffffffff, a.x, offset);
  a.y = __shfl_down_sync(0xffffffff, a.y, offset);
  return *reinterpret_cast<double*>(&a);
}

template <typename T>
__device__ inline T warp_shuffle(T val, int srcLane) {
  return __shfl_sync(0xffffffff, val, srcLane);
}

template <>
__device__ inline double warp_shuffle(double val, int srcLane) {
  int2 a = *reinterpret_cast<int2*>(&val);
  a.x = __shfl_sync(0xffffffff, a.x, srcLane);
  a.y = __shfl_sync(0xffffffff, a.y, srcLane);
  return *reinterpret_cast<double*>(&a);
}

/* 计算softmax(QK^T*scale+Mask)V，block级别,每个warp负责一行，计算完成后更新到o_s里面 */
template <typename T, typename AccumT>
__device__ void block_attentionV1(const T* __restrict__ q_s, const T* __restrict__ k_s, const T* __restrict__ v_s,
                                  AccumT* __restrict__ s_s, T* __restrict__ o_s, AccumT* __restrict__ l_s,
                                  AccumT* __restrict__ m_s, bool is_causal, int Br, int Bc, int d, int q_stride,
                                  int k_stride, int v_stride, int s_stride, int o_stride, int q_row_start,
                                  int k_col_start, int q_len, int k_len, float scale) {
  // 确定当前warp负责哪一行，以及当前线程是当前warp中的第几个
  int warp_idx = threadIdx.x / warpSize;
  int lane_idx = threadIdx.x & (warpSize - 1);
  int num_warps = blockDim.x / warpSize;
  // 外循环遍历所有行，一个warp处理一行
  for (int i = warp_idx; i < Br; i += num_warps) {
    int q_row = q_row_start + i;
    if (q_row >= q_len) {
      continue;
    }

    // m_tile 初始化为负无穷
    AccumT m_tile;
    if constexpr (std::is_same_v<AccumT, float>) {
      m_tile = -CUDART_INF_F;
    } else {
      m_tile = -1.0 / 0.0;
    }

    // 内循环遍历所有列, 一次处理一整列的点积，实际上是读取k_s的一整行
    for (int j = 0; j < Bc; j++) {
      int k_col = k_col_start + j;
      AccumT Sij = 0;  // 使用高精度累加器

      // 一个warp内的一个线程计算一次乘加
      if (k_col < k_len) {
        for (int k = lane_idx; k < d; k += warpSize) {
          int q_offset = i * q_stride + k;
          int k_offset = j * k_stride + k;
          AccumT q_val;
          AccumT k_val;
          if constexpr (std::is_same_v<T, half>) {
            q_val = static_cast<AccumT>(__half2float(q_s[q_offset]));
            k_val = static_cast<AccumT>(__half2float(k_s[k_offset]));
          } else {
            q_val = static_cast<AccumT>(q_s[q_offset]);
            k_val = static_cast<AccumT>(k_s[k_offset]);
          }
          Sij += q_val * k_val;
        }
      }
      // warp内归约得到点积，做完之后lane_id=0的线程拿到了最终结果
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        Sij += warp_shuffle_down(Sij, offset);
      }
      // 一个warp内的0号线程来计算scale和causal mask
      if (lane_idx == 0) {
        // scale: 1/sqrt(d)
        Sij *= static_cast<AccumT>(scale);
        // 越界列直接mask为无效
        if (k_col >= k_len) {
          if constexpr (std::is_same_v<AccumT, float>)
            Sij = -CUDART_INF_F;
          else
            Sij = -1.0 / 0.0;
        } else if (is_causal) {
          int row = q_row;
          int col = k_col;
          if (col > row) {
            if constexpr (std::is_same_v<AccumT, float>)
              Sij = -CUDART_INF_F;
            else
              Sij = -1.0 / 0.0;
          }
        }
        // 将casual(QK^T/scale)的结果写入S
        int s_offset = i * s_stride + j;
        s_s[s_offset] = Sij;  // 存储为 AccumT
        // 更新第i行的m
        if constexpr (std::is_same_v<AccumT, float>) {
          m_tile = fmaxf(m_tile, Sij);
        } else {
          m_tile = fmax(m_tile, Sij);
        }
      }
    }
    // 广播当前行的最大值
    m_tile = warp_shuffle(m_tile, 0);
    AccumT m_old = m_s[i];
    AccumT l_old = l_s[i];
    // 更新当前行的历史最大值
    AccumT m_new;
    if constexpr (std::is_same_v<AccumT, float>) {
      m_new = fmaxf(m_old, m_tile);
    } else {
      m_new = fmax(m_old, m_tile);
    }

    AccumT exp_m_diff;
    if constexpr (std::is_same_v<AccumT, double>) {
      exp_m_diff = exp(m_old - m_new);
    } else {
      exp_m_diff = expf(m_old - m_new);
    }
    AccumT l_new = l_old * exp_m_diff;

    AccumT row_sum = static_cast<AccumT>(0);
    // 一个warp内的每个线程计算各自列的exp(Sij-m_new)
    for (int j = lane_idx; j < Bc; j += warpSize) {
      int k_col = k_col_start + j;
      AccumT s_val = s_s[i * s_stride + j];  // 读取 AccumT
      AccumT exp_val;
      if constexpr (std::is_same_v<AccumT, double>) {
        exp_val = (k_col < k_len) ? exp(s_val - m_new) : 0.0;
      } else {
        exp_val = (k_col < k_len) ? expf(s_val - m_new) : 0.0f;
      }
      s_s[i * s_stride + j] = exp_val;  // 存储 AccumT
      row_sum += exp_val;
    }
    // warp级别规约，计算出每一行的row_sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      row_sum += warp_shuffle_down(row_sum, offset);
    }
    if (lane_idx == 0) {
      l_new += row_sum;
    }
    // 广播l_new，这样所有线程拿到的都是正确的l_new
    l_new = warp_shuffle(l_new, 0);
    // 延迟归一化：P 保持为 exp(S - m_new)，不除以 l_new
    // 更新o_s (deferred normalization): O_unnorm = exp(m_old - m_new) * O_unnorm_old + P @ V
    AccumT o_scale = exp_m_diff;
    // 计算 Pij@Vj
    for (int k = lane_idx; k < d; k += warpSize) {
      AccumT pv = static_cast<AccumT>(0);
      for (int j = 0; j < Bc; j++) {
        int k_col = k_col_start + j;
        if (k_col >= k_len) {
          continue;
        }
        AccumT p_val;
        AccumT v_val;
        p_val = s_s[i * s_stride + j];  // 读取 AccumT
        if constexpr (std::is_same_v<T, half>) {
          v_val = static_cast<AccumT>(__half2float(v_s[j * v_stride + k]));
        } else {
          v_val = static_cast<AccumT>(v_s[j * v_stride + k]);
        }
        pv += p_val * v_val;
      }
      AccumT o_old;
      if constexpr (std::is_same_v<T, half>) {
        o_old = static_cast<AccumT>(__half2float(o_s[i * o_stride + k]));
        o_s[i * o_stride + k] = __float2half_rn(static_cast<float>(o_old * o_scale + pv));
      } else {
        o_old = static_cast<AccumT>(o_s[i * o_stride + k]);
        o_s[i * o_stride + k] = static_cast<T>(o_old * o_scale + pv);
      }
    }
    if (lane_idx == 0) {
      l_s[i] = l_new;
      m_s[i] = m_new;
    }
  }
  __syncthreads();
}