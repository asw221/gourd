
#include <cmath>
#include <iostream>
#include <stdexcept>



template< typename T >
T abseil::standard_logistic::cdf( const T x ) {
  return 1 / (1 + std::exp( -x ));
};



template< typename T >
T abseil::standard_logistic::log_pdf( const T x ) {
  const T inv_ex = std::exp( -x );
  return -x - 2 * std::log( 1 + inv_ex );
};



template< typename T >
T abseil::standard_logistic::pdf( const T x ) {
  const T inv_ex = std::exp( -x );
  return inv_ex / ( (1 + inv_ex) * (1 + inv_ex) );
};


template< typename T >
T abseil::standard_logistic::quantile( const T p ) {
  T Q = qmax_v<T>;
  if ( p < static_cast<T>(0) || p > static_cast<T>(1) ) {
    throw std::domain_error(
      "standard_logistic::quantile defined on [0, 1]");
  }
  if ( p == static_cast<T>(0) || p == static_cast<T>(1) ) {
#ifndef DNDEBUG
    std::cerr << "WARNING: standard_logistic::quantile: "
	      << "probability numerically 0 or 1\n";
#endif
    if ( p == static_cast<T>(0) ) {
      Q = -Q;
    }
  }
  else {
    Q = std::log( p / (1 - p) );
  }
  return Q;
};

