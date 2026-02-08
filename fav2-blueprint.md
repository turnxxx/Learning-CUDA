```c++
// --- 1. 寄存器申请 (Warp 级分块) ---
// 每个线程只持有矩阵的一小块碎片
// Tr = Br / Warp_Rows / Lane_Rows
// Tc = Bc / Warp_Cols / Lane_Cols
float r_S[Tr][Tc];      // 存放 Scores (S) 和 Probabilities (P)
float r_O[Tr][HeadDim]; // 存放最终输出 O
float r_m[Tr];          // 存放每行的 Max
float r_l[Tr];          // 存放每行的 Sum

// 初始化寄存器
clear(r_O);
fill(r_m, -INFINITY);
fill(r_l, 0);

// --- 2. Prologue (流水线启动) ---
// 启动第 0 块 K 和 V 的异步加载
// 注意：这里假设 sK 和 sV 只能存一块数据 (单缓冲)
cp_async(sK, Global_K[0]);
cp_async(sV, Global_V[0]); // 实际中 V 的加载可能更晚，这里简化
cp_async_commit_group();   // Group 0: 包含 K[0], V[0]

// --- 3. Main Loop (遍历 K/V Blocks) ---
for (int i = 0; i < Num_KV_Blocks; ++i) {

    // ==========================================
    // 阶段 A: 计算 S = Q * K[i]^T
    // ==========================================

    // [Wait]: 等待 K[i] 到位
    // 如果是 i=0，等的是 Prologue 发起的。
    // 如果是 i>0，等的是上一轮阶段 B 发起的。
    cp_async_wait_group(0); 
    __syncthreads();

    // [Compute]: GEMM 1 (Warp 分块计算)
    // 每个线程计算自己负责的那部分 r_S
    // r_S += r_Q_frag * r_K_frag
    // 注意：sV 此时闲置，存着 V[i]
    gemm_warp_tiled(r_S, sQ, sK); 

    // ==========================================
    // 阶段 B: 在线 Softmax 更新 (部分)
    // ==========================================
    
    // 1. 局部 Max 更新
    // 线程内找最大值 -> Warp 内 Shuffle 交换 -> 更新 r_m
    update_row_max_with_shuffle(r_S, r_m);

    // 2. 计算 P = exp(S - m)
    // 直接在寄存器 r_S 上原地修改
    for (int r=0; r<Tr; ++r)
        for (int c=0; c<Tc; ++c)
            r_S[r][c] = exp(r_S[r][c] - r_m[r]);

    // ==========================================
    // 阶段 C: 流水线预取 (关键点!)
    // ==========================================
    
    // 此时 S 算完了，K[i] 没用了！
    // 我们可以用 K[i+1] 覆盖 sK！
    // 这个加载操作将与下面的 GEMM 2 并行执行
    if (i < Num_KV_Blocks - 1) {
        __syncthreads(); // 确保所有线程都读完了 K[i]
        cp_async(sK, Global_K[i+1]);
        // 注意：V[i+1] 的加载策略有多种，最激进的是复用 sV
        // 但为了简化，这里假设 sK 和 sV 是独立的单缓冲
        cp_async(sV, Global_V[i+1]); 
        cp_async_commit_group(); 
    }

    // ==========================================
    // 阶段 D: 计算 O += P * V[i]
    // ==========================================

    // [Compute]: GEMM 2
    // r_O += r_S * sV
    // 此时 DMA 引擎正在后台把 K[i+1] 搬运到 sK
    gemm_warp_tiled(r_O, r_S, sV);
    
    // ==========================================
    // 阶段 E: Softmax 归一化统计量更新
    // ==========================================
    
    // 更新 r_l (Sum)
    update_row_sum_with_shuffle(r_S, r_l);
    
    // Rescale r_O (根据新的 Max/Sum 修正旧结果)
    rescale_output(r_O, r_m, r_l);
}

// --- 4. Epilogue (写回) ---
// 此时 r_O 还在寄存器里
// 1. 最终归一化: r_O /= r_l
// 2. 寄存器 -> Shared Memory (为了合并写)
//    这里复用 sQ 的空间！
store_regs_to_smem(sQ, r_O);
__syncthreads();

// 3. Shared Memory -> Global Memory
store_smem_to_global(Global_O, sQ);
```
关键点解析

1. float r_S[Tr][Tc]:
这就是寄存器驻留的体现。我们没有申请巨大的 S[Br][Bc]，而是申请了每个线程私有的小碎片。

2. gemm_warp_tiled:
这个函数内部包含多重循环（K 维度的循环），每次从 Shared Memory 加载一小块数据到临时寄存器，然后做外积累加到 r_S。
update_row_max_with_shuffle:
这是Warp 级分块的必修课。因为一行数据被切分到了不同的线程，所以必须用 __shfl_xor_sync 来进行跨线程通信，算出全局 Max。
3. cp_async 插入位置:
我们在计算 GEMM 2 (P * V) 之前，发起了下一轮 $K$ 和 $V$ 的加载。
    
    安全性：此时 $K[i]$ 已经用完了，可以覆盖。

    并行性：GEMM 2 是计算密集型的，正好用来掩盖 Global Memory 的延迟。
