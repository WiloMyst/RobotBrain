#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <optional>
#include <atomic>

namespace engine {
namespace infra {

/// 固定大小线程池
/// 核心特性:enqueue 返回 std::optional<future>,队列满时返回 nullopt
/// 调用方可据此做背压降级,而非无限排队导致 OOM
class ThreadPool {
public:
    ThreadPool(size_t threads, size_t max_queue = 64);
    ~ThreadPool();

    /// 提交任务到线程池
    /// @return future 若队列未满;nullopt 若队列已满(背压保护)
    std::optional<std::future<void>> enqueue(std::function<void()> task);

    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mtx_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;

    std::atomic<bool> stop_{false};
    size_t max_queue_;
};

} // namespace infra
} // namespace engine
