#ifndef LOG_H
#define LOG_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// ============================================================================
// CUDA 错误检查宏
// ============================================================================

/**
 * @brief 检查 CUDA 调用是否成功，失败时打印错误并退出
 * @param call CUDA API 调用
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief 检查 CUDA 调用是否成功，失败时打印错误并返回错误码
 * @param call CUDA API 调用
 */
#define CHECK_CUDA_ERR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

/**
 * @brief 检查 CUDA 内核启动后的错误（包括同步）
 * @note 内核启动是异步的，需要同步才能捕获错误
 */
#define CHECK_KERNEL_LAUNCH() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Kernel Launch Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief 检查 CUDA 内核启动错误（不退出，返回错误码）
 */
#define CHECK_KERNEL_LAUNCH_ERR() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Kernel Launch Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

/**
 * @brief 检查并获取最后一个 CUDA 错误
 */
#define CHECK_LAST_CUDA_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// 日志宏
// ============================================================================

/**
 * @brief 调试日志（仅在 DEBUG 模式下输出）
 */
#ifdef DEBUG
    #define LOG_DEBUG(fmt, ...) \
        fprintf(stdout, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(fmt, ...) ((void)0)
#endif

/**
 * @brief 信息日志
 */
#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)

/**
 * @brief 警告日志
 */
#define LOG_WARN(fmt, ...) \
    fprintf(stderr, "[WARN] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

/**
 * @brief 错误日志
 */
#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// ============================================================================
// 断言宏
// ============================================================================

/**
 * @brief CUDA 设备断言（在设备代码中使用）
 */
#define CUDA_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            printf("Assertion failed at %s:%d\n", __FILE__, __LINE__); \
            asm("trap;"); \
        } \
    } while (0)

/**
 * @brief 主机端断言（带消息）
 */
#define ASSERT_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "[Assertion Failed] %s:%d: %s\n", __FILE__, __LINE__, msg); \
            assert(condition); \
        } \
    } while (0)

// ============================================================================
// 内存检查宏
// ============================================================================

/**
 * @brief 检查指针是否为空
 */
#define CHECK_PTR(ptr) \
    do { \
        if ((ptr) == nullptr) { \
            fprintf(stderr, "[Error] %s:%d: Null pointer\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief 检查 CUDA 内存分配是否成功
 */
#define CHECK_CUDA_MALLOC(ptr, size) \
    do { \
        CHECK_CUDA(cudaMalloc(&(ptr), (size))); \
        if ((ptr) == nullptr) { \
            fprintf(stderr, "[Error] %s:%d: cudaMalloc returned null pointer\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// 性能计时宏
// ============================================================================

/**
 * @brief 创建 CUDA 事件用于计时
 */
#define CREATE_CUDA_EVENTS(start, stop) \
    cudaEvent_t start, stop; \
    CHECK_CUDA(cudaEventCreate(&start)); \
    CHECK_CUDA(cudaEventCreate(&stop))

/**
 * @brief 记录开始时间
 */
#define RECORD_START(start) CHECK_CUDA(cudaEventRecord(start))

/**
 * @brief 记录结束时间并计算耗时（毫秒）
 */
#define RECORD_STOP_ELAPSED(start, stop, elapsed_ms) \
    do { \
        CHECK_CUDA(cudaEventRecord(stop)); \
        CHECK_CUDA(cudaEventSynchronize(stop)); \
        CHECK_CUDA(cudaEventElapsedTime(&(elapsed_ms), start, stop)); \
    } while (0)

/**
 * @brief 销毁 CUDA 事件
 */
#define DESTROY_CUDA_EVENTS(start, stop) \
    do { \
        CHECK_CUDA(cudaEventDestroy(start)); \
        CHECK_CUDA(cudaEventDestroy(stop)); \
    } while (0)

#endif // LOG_H