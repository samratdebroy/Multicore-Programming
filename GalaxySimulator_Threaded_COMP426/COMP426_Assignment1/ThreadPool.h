#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace {
	template <typename T>
	class ConcurrentQueue {
	public:
		void push(T const & value) {
			std::unique_lock<std::mutex> lock(mutex_);
			q_.push(value);
		}
		// only for integral types
		bool pop(T & v) {
			std::unique_lock<std::mutex> lock(mutex_);
			if (q_.empty())
				return false;
			v = std::move(q_.front());
			q_.pop();
			return true;
		}
		bool empty() {
			std::unique_lock<std::mutex> lock(mutex_);
			return q_.empty();
		}
	private:
		std::queue<T> q_;
		std::mutex mutex_;
	};
}


class ThreadPool {
private:
	std::vector<std::unique_ptr<std::thread>> threads_; 
	ConcurrentQueue<std::function<void()>> queue_;
	std::atomic<bool> isClosing_;
	std::mutex mutex_;
	std::condition_variable conditionalLock_;


	class ThreadWorker {
	private:
		int id_;
		ThreadPool* pool_;
	public:
		ThreadWorker(ThreadPool * pool, const int id)
			: pool_(pool), id_(id) {}

		// ThreadWorker will continously execute tasks assigned to the queue
		// This will be called by the thread once its assigned this worker
		void operator()() {
			std::function<void()> func;
			bool popped;
			//  Keep executing tasks until the threadpool closes
			while (!pool_->isClosing_) {
				{
					std::unique_lock<std::mutex> lock(pool_->mutex_);
					// Wait until the task queue isn't empty before resuming
					if (pool_->queue_.empty()) {
						pool_->conditionalLock_.wait(lock);
					}
					popped = pool_->queue_.pop(func);
				}
				if (popped) {
					func();
				}
			}
		}

	};

public:

	ThreadPool() : isClosing_(false) {}
	ThreadPool(const int threadNum) : isClosing_(false)
	{
		init(threadNum);
	}

	~ThreadPool()
	{
		// Exit all threads
		close();
	}

	void init(const int threadNum)
	{
		threads_.reserve(threadNum);
		for (int i = 0; i < threadNum; ++i)
		{
			threads_.emplace_back(std::make_unique<std::thread>(ThreadWorker(this, i)));
		}
	}

	void close() {
		isClosing_ = true;
		conditionalLock_.notify_all(); // Release all waiting locks
		// Join all threads in the pool
		for (auto& thread : threads_)
		{
			if (thread->joinable())
			{
				thread->join();
			}
		}
	}

	// Push a function to be added to the task queue for the threads to execute
	// Use a variadic template to make sure we support functions with arguments
	template<typename F, typename...Args>
	auto push(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
		// Bind the arguments to the function
		std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
		// Packaged task allows you to asynchronously execute functions and shared ptr makes allows copy construct
		auto task_package = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);
		// Wrap packaged task into void function so it's a consistent data type for the queue
		std::function<void()> wrapper_func = [task_package]() {
			(*task_package)();
		};

		// Enqueue the wrapper function
		queue_.push(wrapper_func);

		// Wake up one waiting thread
		conditionalLock_.notify_one();

		// Return future from promise
		return task_package->get_future();
	}

private:
	// Non-copyable and Non-moveable
	ThreadPool(const ThreadPool& rhs) = delete;
	ThreadPool(const ThreadPool&& rhs) = delete;
	// Non-assignable.
	ThreadPool& operator=(const ThreadPool& rhs) = delete;
	ThreadPool& operator=(const ThreadPool&& rhs) = delete;

};