// https://stackoverflow.com/questions/12883493/timing-the-execution-of-statements-c
// https://github.com/CrikeeIP/Stopwatch/blob/master/include/stopwatch/Stopwatch.hpp
#pragma once
#include <chrono>

class DIYTimer
{
private:
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_pt;

    time_pt start_time;
    time_pt finish_time;

    unsigned int elapsed_time_ms; // = 0;

public:
    DIYTimer()
    {
        start();
    }

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    unsigned int finish()
    {
        finish_time = std::chrono::high_resolution_clock::now();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time);

        elapsed_time_ms = milliseconds.count();

        return elapsed_time_ms;
    }
};