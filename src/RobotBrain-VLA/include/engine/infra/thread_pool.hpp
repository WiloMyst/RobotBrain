#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>
#include <optional>

namespace engine { 
namespace infra { 

class ThreadPool {
public:
    // 构造函数：增加 max_queue_size 参数
    ThreadPool(size_t threads, size_t max_queue_size = 1000) 
        : stop(false), max_queue_size_(max_queue_size) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    // 企业级改造：返回 std::optional。如果队列满了，返回空，触发业务熔断
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::optional<std::future<typename std::result_of<F(Args...)>::type>> 
    {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // 背压检查：如果队列已满，直接拒绝
            if(tasks.size() >= max_queue_size_) {
                return std::nullopt; 
            }
            if(stop) throw std::runtime_error("ThreadPool stopped");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    size_t max_queue_size_; // 队列上限
};

} // namespace infra
} // namespace engine