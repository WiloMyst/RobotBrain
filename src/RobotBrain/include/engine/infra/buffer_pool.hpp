#pragma once
#include <vector>
#include <memory>
#include <stack>
#include <mutex>
#include <atomic>
#include <functional>

namespace engine {
namespace infra {

/// 对象内存池:预分配 + RAII 归还
/// 消除推理热路径上的堆分配,配合 Ort::Value::CreateTensor 实现零拷贝
///
/// 使用方式:
///   auto buf = pool.Acquire();      // 借出一块 float 缓冲区
///   // ... 使用 buf->data() ...
///   // buf 析构时自动归还池中,无需手动 Release
template <typename T>
class BufferPool {
public:
    /// 预分配 count 块,每块 size 个 T 元素
    BufferPool(size_t count, size_t size)
        : element_size_(size), alive_(true) {
        for (size_t i = 0; i < count; ++i) {
            pool_.push(std::make_unique<std::vector<T>>(size));
        }
    }

    ~BufferPool() {
        alive_.store(false);
    }

    /// 借出一块缓冲区,析构时自动归还
    /// @return unique_ptr<vector<T>>,自定义删除器归还到池中
    std::unique_ptr<std::vector<T>, std::function<void(std::vector<T>*)>> Acquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pool_.empty()) {
            // 池耗尽:临时分配新块,不归还到池中
            return std::unique_ptr<std::vector<T>, std::function<void(std::vector<T>*)>>(
                new std::vector<T>(element_size_),
                [](std::vector<T>* p) { delete p; }
            );
        }
        auto raw = pool_.top().release();
        pool_.pop();
        // 自定义 deleter:归还到池中
        return std::unique_ptr<std::vector<T>, std::function<void(std::vector<T>*)>>(
            raw,
            [this](std::vector<T>* p) {
                if (alive_.load()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    pool_.push(std::unique_ptr<std::vector<T>>(p));
                } else {
                    delete p;
                }
            }
        );
    }

private:
    size_t element_size_;
    std::stack<std::unique_ptr<std::vector<T>>> pool_;
    std::mutex mtx_;
    std::atomic<bool> alive_;
};

} // namespace infra
} // namespace engine
