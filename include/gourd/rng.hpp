
#include <random>


#ifndef _GOURD_RNG_
#define _GOURD_RNG_

namespace gourd {
namespace __1 {
  std::mt19937 __urng;
}
// namespace __1


  /*! RNG access */
  constexpr std::mt19937& urng() { return gourd::__1::__urng; };

  /*! Set RNG seed */
  inline void set_urng_seed( const unsigned seed ) {
    gourd::__1::__urng.seed(seed);
  };
  
}
// namespace gourd

#endif  // _GOURD_RNG_

