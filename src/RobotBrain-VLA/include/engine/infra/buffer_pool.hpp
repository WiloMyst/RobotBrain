#pragma once
#include <vector>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace engine {
namespace infra {

/**
 * @brief 高性能线程安全缓冲池 (企业级 RAII 实现)
 * 适用于 AI 推理时的高频 Tensor 内存复用，彻底消灭 Heap Allocation
 */
template <typename T>
class BufferPool {
public:
    using BufferType = std::vector<T>;

    // ====================================================================
    // 核心黑魔法：自定义智能指针删除器 (Custom Deleter)
    // 作用：当智能指针生命周期结束时，不调用 delete 销毁内存，
    // 而是自动把内存块“还”回池子里！
    // ====================================================================
    class PoolDeleter {
    public:
        explicit PoolDeleter(BufferPool* pool = nullptr) : pool_(pool) {}
        void operator()(BufferType* ptr) const {
            if (pool_) {
                pool_->Release(ptr);
            } else {
                delete ptr; // 兜底：如果池子没了，再真正销毁
            }
        }
    private:
        BufferPool* pool_;
    };

    // 定义我们专属的“带自动回收功能的唯一指针”
    using BufferPtr = std::unique_ptr<BufferType, PoolDeleter>;

    // ====================================================================
    // 构造与预分配
    // ====================================================================
    BufferPool(size_t pool_size, size_t buffer_capacity) 
        : buffer_capacity_(buffer_capacity) {
        
        spdlog::info(" [BufferPool] 正在预分配物理内存... 容量: {} 块, 每块大小: {}", 
                     pool_size, buffer_capacity);

        for (size_t i = 0; i < pool_size; ++i) {
            auto buf = new BufferType();
            buf->resize(buffer_capacity_, T()); // 直接撑开物理内存并清零
            free_buffers_.push_back(buf);
        }
    }

    ~BufferPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto buf : free_buffers_) {
            delete buf;
        }
        free_buffers_.clear();
    }

    // ====================================================================
    // 借出内存 (Acquire)
    // ====================================================================
    BufferPtr Acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_buffers_.empty()) {
            // 极限并发下的背压警告：池子被借空了！
            // 工业级做法是动态降级：临时 new 一块给业务用，保证系统不挂
            spdlog::warn(" [BufferPool] 缓冲池已耗尽！触发临时动态分配，请注意排查并发压力！");
            auto temp_buf = new BufferType();
            temp_buf->resize(buffer_capacity_, T());
            return BufferPtr(temp_buf, PoolDeleter{this});
        }

        // 从池中取出一块
        BufferType* ptr = free_buffers_.back();
        free_buffers_.pop_back();
        
        // 包装成 RAII 智能指针返回
        return BufferPtr(ptr, PoolDeleter{this});
    }

private:
    // ====================================================================
    // 归还内存 (Release) - 仅供 Deleter 内部调用
    // ====================================================================
    void Release(BufferType* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_buffers_.push_back(ptr);
    }

    std::vector<BufferType*> free_buffers_; // 空闲内存块列表 (栈结构)
    size_t buffer_capacity_;                // 每块内存的固定大小
    std::mutex mutex_;                      // 保证多线程安全借还
};

} // namespace infra
} // namespace engine