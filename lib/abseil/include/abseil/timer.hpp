
#include <chrono>
// #include <iostream>


#ifndef _ABSEIL_TIMER_
#define _ABSEIL_TIMER_

namespace abseil {

  class timer {
    using clock  = std::chrono::high_resolution_clock;
    using time_t = std::chrono::time_point<clock>;

  public:
    using nanoseconds  = std::chrono::nanoseconds;
    using microseconds = std::chrono::microseconds;
    using milliseconds = std::chrono::milliseconds;
    using seconds      = std::chrono::seconds;
    using minutes      = std::chrono::minutes;
    
    static void start() { start_ = clock::now(); };
    static void stop()  { stop_  = clock::now(); };

    template< typename Units = microseconds >
    static double diff() {
      return (double)std::chrono::duration_cast<Units>
	(stop_ - start_).count();
    };
    
  private:
    static time_t start_;
    static time_t stop_;
  };

};


abseil::timer::time_t
abseil::timer::start_ = abseil::timer::clock::now();

abseil::timer::time_t
abseil::timer::stop_  = abseil::timer::clock::now();

#endif  // _ABSEIL_TIMER_
