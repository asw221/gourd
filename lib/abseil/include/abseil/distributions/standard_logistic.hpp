

#ifndef _ABSEIL_STANDARD_LOGISTIC_DISTRIBUTION_
#define _ABSEIL_STANDARD_LOGISTIC_DISTRIBUTION_


namespace abseil {


  class standard_logistic {
  public:
    template< typename T > static inline constexpr T qmax_v = 36.72;

    template< typename T > static T cdf( const T x );
    template< typename T > static T log_pdf( const T x );
    template< typename T > static T pdf( const T x );    
    template< typename T > static T quantile( const T p );
  };
  

};


#include "abseil/distributions/standard_logistic_def.inl"

#endif  // _ABSEIL_STANDARD_LOGISTIC_DISTRIBUTION_
