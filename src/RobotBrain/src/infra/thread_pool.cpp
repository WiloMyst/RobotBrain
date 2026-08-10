#include "engine/infra/thread_pool.hpp"
#include <spdlog/spdlog.h>

namespace engine {
namespace infra {

ThreadPool::ThreadPool(size_t threads, size_t max_queue)
    : max_queue_(max_queue) {
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_not_empty_.wait(lock, [this]() {
                        return stop_.load() || !tasks_.empty();
                    });

                    if (stop_.load() && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                cv_not_full_.notify_one();

                task();
            }
        });
    }
    spdlog::info("ThreadPool created: {} threads, max_queue={}", threads, max_queue);
}

std::optional<std::future<void>> ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if (tasks_.size() >= max_queue_) {
            // 背压:队列已满,拒绝入队
            return std::nullopt;
        }
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();

        tasks_.emplace([p = std::move(promise), t = std::move(task)]() {
            try {
                t();
                p->set_value();
            } catch (...) {
                p->set_exception(std::current_exception());
            }
        });

        cv_not_empty_.notify_one();
        return future;
    }
}

void ThreadPool::shutdown() {
    stop_.store(true);
    cv_not_empty_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

ThreadPool::~ThreadPool() {
    if (!stop_.load()) {
        shutdown();
    }
}

} // namespace infra
} // namespace engine
