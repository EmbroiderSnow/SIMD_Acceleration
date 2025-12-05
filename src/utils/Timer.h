#pragma once
#include <chrono>

class Timer {
public:
    Timer() {
        reset();
    }

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time;
        return diff.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};