#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "./log.h"
/* 计算softmax(QK^T*scale+Mask)V，block级别,每个warp负责一行，计算完成后更新到o_s里面 */
template <typename T, typename AccumT>
__device__ void block_attentionV1(const T* __restrict__ q_s, const T* __restrict__ k_s, const T* __restrict__ v_s,
                                  float* __restrict__ s_s, T* __restrict__ o_s, AccumT* __restrict__ l_s,
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
    float m_tile = -CUDART_INF_F;
    // 内循环遍历所有列, 一次处理一整列的点积
    for (int j = 0; j < Bc; j++) {
      int k_col = k_col_start + j;
      float Sij = 0.0f;  // 每个线程的一个Sij
      // 一个warp内的一个线程计算一次乘加
      if (k_col < k_len) {
        for (int k = lane_idx; k < d; k += warpSize) {
          int q_offset = i * q_stride + k;
          int k_offset = j * k_stride + k;
          float q_val;
          float k_val;
          if constexpr (std::is_same_v<T, half>) {
            q_val = __half2float(q_s[q_offset]);
            k_val = __half2float(k_s[k_offset]);
          } else {
            q_val = static_cast<float>(q_s[q_offset]);
            k_val = static_cast<float>(k_s[k_offset]);
          }
          Sij += q_val * k_val;
        }
      }
      // warp内归约得到点积
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        Sij += __shfl_down_sync(0xffffffff, Sij, offset);
      }
      // 一个warp内的0号线程来计算scale和causal mask
      if (lane_idx == 0) {
        // scale: 1/sqrt(d)
        Sij *= scale;
        // 越界列直接mask为无效
        if (k_col >= k_len) {
          Sij = -CUDART_INF_F;
        } else if (is_causal) {
          int row = q_row;
          int col = k_col;
          if (col > row) {
            Sij = -CUDART_INF_F;
          }
        }
        // 将casual(QK^T/scale)的结果写入S
        int s_offset = i * s_stride + j;
        s_s[s_offset] = Sij;
        // 更新第i行的m
        m_tile = fmaxf(m_tile, Sij);
      }
    }
    // 广播当前行的最大值
    m_tile = __shfl_sync(0xffffffff, m_tile, 0);
    AccumT m_old = m_s[i];
    AccumT l_old = l_s[i];
    // 更新当前行的历史最大值
    AccumT m_new = static_cast<AccumT>(fmax(static_cast<float>(m_old), m_tile));
    AccumT l_new =
        l_old * (std::is_same_v<AccumT, double> ? static_cast<AccumT>(exp(m_old - m_new)) : expf(m_old - m_new));
    AccumT row_sum = static_cast<AccumT>(0);
    // 一个warp内的每个线程计算各自列的exp(Sij-m_new)
    for (int j = lane_idx; j < Bc; j += warpSize) {
      int k_col = k_col_start + j;
      float s_val = s_s[i * s_stride + j];
      float exp_val = (k_col < k_len) ? expf(s_val - m_new) : 0.0f;
      s_s[i * s_stride + j] = exp_val;
      row_sum += static_cast<AccumT>(exp_val);
    }
    // warp级别规约，计算出每一行的row_sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }
    if (lane_idx == 0) {
      l_new += row_sum;
    }
    // 广播l_new，这样所有线程拿到的都是正确的l_new
    l_new = __shfl_sync(0xffffffff, l_new, 0);
    // 再次遍历列，更新P (float精度)
    for (int j = lane_idx; j < Bc; j += warpSize) {
      s_s[i * s_stride + j] = s_s[i * s_stride + j] / static_cast<float>(l_new);
    }
    // 更新o_s: O = (l_old * exp(m_old - m_new) * O + P @ V) / l_new
    AccumT o_scale =
        (l_old * (std::is_same_v<AccumT, double> ? static_cast<AccumT>(exp(m_old - m_new)) : expf(m_old - m_new))) /
        l_new;
    // 计算 Pij@Vj
    for (int k = lane_idx; k < d; k += warpSize) {
      AccumT pv = static_cast<AccumT>(0);
      for (int j = 0; j < Bc; j++) {
        int k_col = k_col_start + j;
        if (k_col >= k_len) {
          continue;
        }
        float p_val;
        float v_val;
        p_val = (s_s[i * s_stride + j]);
        if constexpr (std::is_same_v<T, half>) {
          v_val = __half2float(v_s[j * v_stride + k]);
        } else {
          v_val = static_cast<float>(v_s[j * v_stride + k]);
        }
        pv += static_cast<AccumT>(p_val * v_val);
      }
      float o_old;
      if constexpr (std::is_same_v<T, half>) {
        o_old = __half2float(o_s[i * o_stride + k]);
        o_s[i * o_stride + k] = __float2half_rn(o_old * static_cast<float>(o_scale) + static_cast<float>(pv));
      } else {
        o_old = static_cast<float>(o_s[i * o_stride + k]);
        o_s[i * o_stride + k] = static_cast<T>(o_old * static_cast<float>(o_scale) + static_cast<float>(pv));
      }
    }
    if (lane_idx == 0) {
      l_s[i] = l_new;
      m_s[i] = m_new;
    }
  }
  __syncthreads();
}