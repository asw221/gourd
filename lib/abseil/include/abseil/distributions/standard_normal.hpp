

#ifndef _ABSEIL_STANDARD_NORMAL_DISTRIBUTION_
#define _ABSEIL_STANDARD_NORMAL_DISTRIBUTION_


namespace abseil {


  /*! Standard normal distribution functions
   *
   * Quantile algorithm from:
   * 
   * Wichura, Michael J. "Algorithm AS 241: The percentage points of 
   *   the normal distribution." Journal of the Royal Statistical 
   *   Society. Series C (Applied Statistics) 37.3 (1988): 477-484.
   *
   */
  class standard_normal {
  public:
    template< typename T > static inline constexpr T qmax_v = 8.160708;

    template< typename T > static T cdf( const T x );
    template< typename T > static T log_pdf( const T x );
    template< typename T > static T pdf( const T x );    
    template< typename T > static T quantile( const T p );
  };
  

};



#include "abseil/distributions/standard_normal_def.inl"


#endif  // _ABSEIL_STANDARD_NORMAL_DISTRIBUTION_
