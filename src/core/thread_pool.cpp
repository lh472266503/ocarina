//
// Created by Zero on 2023/7/10.
//

#include "thread_pool.h"
#include "logging.h"

#if (!defined(__clang_major__) || __clang_major__ >= 14) && defined(__cpp_lib_barrier)
#define OC_USE_STD_BARRIER
#endif

#ifdef OC_USE_STD_BARRIER
#include <barrier>
#endif

namespace ocarina {

namespace detail {

[[nodiscard]] static auto &is_worker_thread() noexcept {
    static thread_local auto is_worker = false;
    return is_worker;
}

[[nodiscard]] static auto &worker_thread_index() noexcept {
    static thread_local auto id = 0u;
    return id;
}

static inline void check_not_in_worker_thread(std::string_view f) noexcept {
    if (is_worker_thread()) [[unlikely]] {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        OC_ERROR_FORMAT(
            "Invoking ThreadPool::{}() "
            "from worker thread {}.",
            f, oss.str());
    }
}

}// namespace detail

#ifdef OC_USE_STD_BARRIER
struct Barrier : std::barrier<> {
    using std::barrier<>::barrier;
};
#else
// reference: https://github.com/yohhoy/yamc/blob/master/include/yamc_barrier.hpp
class Barrier {
private:
    uint _n;
    uint _counter;
    uint _phase;
    std::condition_variable _cv;
    std::mutex _mutex;

public:
    explicit Barrier(uint n) noexcept
        : _n{n}, _counter{n}, _phase{0u} {}
    void arrive_and_wait() noexcept {
        std::unique_lock lock{_mutex};
        auto arrive_phase = _phase;
        if (--_counter == 0u) {
            _counter = _n;
            _phase++;
            _cv.notify_all();
        }
        while (_phase <= arrive_phase) {
            _cv.wait(lock);
        }
    }
};
#endif

ThreadPool::ThreadPool(size_t num_threads) noexcept : _should_stop{false} {
    if (num_threads == 0u) {
        num_threads = std::max(
            std::thread::hardware_concurrency(), 1u);
    }
    _dispatch_barrier = make_unique<Barrier>(num_threads);
    _synchronize_barrier = make_unique<Barrier>(num_threads + 1u /* main thread */);
    _threads.reserve(num_threads);
    for (auto i = 0u; i < num_threads; i++) {
        _threads.emplace_back(std::thread{[this, i] {
            detail::is_worker_thread() = true;
            detail::worker_thread_index() = i;
            for (;;) {
                std::unique_lock lock{_mutex};
                _cv.wait(lock, [this] { return !_tasks.empty() || _should_stop; });
                if (_should_stop && _tasks.empty()) [[unlikely]] { break; }
                auto task = std::move(_tasks.front());
                _tasks.pop();
                lock.unlock();
                task();
            }
        }});
    }
    OC_INFO_FORMAT("Created thread pool with {} thread{}.",
                   num_threads, num_threads == 1u ? "" : "s");
}

void ThreadPool::barrier() noexcept {
    detail::check_not_in_worker_thread("barrier");
    _dispatch_all([this] { _dispatch_barrier->arrive_and_wait(); });
}

void ThreadPool::synchronize() noexcept {
    detail::check_not_in_worker_thread("synchronize");
    while (task_count() != 0u) {
        _dispatch_all([this] { _synchronize_barrier->arrive_and_wait(); });
        _synchronize_barrier->arrive_and_wait();
    }
}

void ThreadPool::_dispatch(std::function<void()> task) noexcept {
    {
        std::scoped_lock lock{_mutex};
        _tasks.emplace(std::move(task));
    }
    _cv.notify_one();
}

void ThreadPool::_dispatch_all(std::function<void()> task, size_t max_threads) noexcept {
    {
        std::scoped_lock lock{_mutex};
        for (auto i = 0u; i < std::min(_threads.size(), max_threads) - 1u; i++) {
            _tasks.emplace(task);
        }
        _tasks.emplace(std::move(task));
    }
    _cv.notify_all();
}

ThreadPool::~ThreadPool() noexcept {
    {
        std::scoped_lock lock{_mutex};
        _should_stop = true;
    }
    _cv.notify_all();
    for (auto &&t : _threads) { t.join(); }
}

ThreadPool &ThreadPool::instance() noexcept {
    static ThreadPool pool;
    return pool;
}

uint ThreadPool::task_count() const noexcept {
    return _task_count.load();
}

uint ThreadPool::worker_thread_index() noexcept {
    OC_ASSERT(detail::is_worker_thread());
    return detail::worker_thread_index();
}

}// namespace ocarina