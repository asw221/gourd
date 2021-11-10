
#include <array>
#include <cmath>
#include <stdexcept>

#include "abseil/math.hpp"


template< typename T >
T abseil::standard_normal::cdf( const T x ) {
  return static_cast<T>(0.5) * std::erfc( -x * num::inv_sqrt2_v<T> );
};


template< typename T >
T abseil::standard_normal::log_pdf( const T x ) {
  return std::log( num::inv_sqrt2pi_v<T> ) -
    static_cast<T>(0.5) * x * x;
};


template< typename T >
T abseil::standard_normal::pdf( const T x ) {
  return num::inv_sqrt2pi_v<T> *
    std::exp( -static_cast<T>(0.5) * x * x );
};


template< typename T >
T abseil::standard_normal::quantile( const T p ) {
  static const T Split_Q = 0.425;
  static const T Split_R = 5.0;
  static const T Split_Qsq = 0.180625;
  static const T Wichura_Constant_2 = 1.6;
  
  static const std::array<T, 8> Wichura_A{
    3.3871328727963996080e0,   1.3314166789178437745e2,
      1.9715909503065514427e3, 1.3731693765509461125e4,
      4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3
      };

  static const std::array<T, 8> Wichura_B{
    1.0,                       4.2313330701600911252e1,
      6.8718700749205790830e2, 5.3941960214247511077e3,
      2.1213794301586595867e4, 3.9307895800092710610e4,
      2.8729085735721942674e4, 5.2264952788528545610e3
      };

  static const std::array<T, 8> Wichura_C{
    1.42343711074968357734e0,   4.63033784615654529590e0,
      5.76949722146069140550e0, 3.64784832476320460504e0,
      1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4
      };

  static const std::array<T, 8> Wichura_D{
    1.0,                         2.05319162663775882187e0,
      1.67638483018380384940e0,  6.89767334985100004550e-1,
      1.48103976427480074590e-1, 1.51986665636164571966e-2,
      5.47593808499534494600e-4, 1.05075007164441684324e-9
      };

  static const std::array<T, 8> Wichura_E{
    6.65790464350110377720e0,    5.46378491116411436990e0,
      1.78482653991729133580e0,  2.96560571828504891230e-1,
      2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7
      };

  static const std::array<T, 8> Wichura_F{
    1.0,                         5.99832206555887937690e-1,
      1.36929880922735805310e-1, 1.48753612908506148525e-2,
      7.86869131145613259100e-4, 1.84631831751005468180e-5,
      1.42151175831644588870e-7, 2.04426310338993978564e-15
      };
  
  const T q = p - static_cast<T>( 0.5 );
  typename std::array<T, 8>::const_reverse_iterator nit, dit;
  T quant = 0;
  T r, numer, denom;

  
  if ( p < static_cast<T>(0) || p > static_cast<T>(1) ) {
    throw std::domain_error(
      "standard_normal::quantile defined on [0, 1]");
  }
#ifndef DNDEBUG
  if ( p == static_cast<T>(0) || p == static_cast<T>(1) ) {
    std::cerr << "WARNING: standard_normal::quantile: "
	      << "probability numerically 0 or 1\n";
  }
#endif
  
  if ( std::abs( q ) <= Split_Q ) {
    r = Split_Qsq - q * q;
    numer = Wichura_A.back();
    denom = Wichura_B.back();
    for ( nit = Wichura_A.rbegin() + 1, dit = Wichura_B.rbegin() + 1;
	  nit != Wichura_A.rend(); nit++, dit++ ) {
      numer = numer * r + (*nit);
      denom = denom * r + (*dit);
    }
    quant = q * numer / denom;
  }
  else {
    r = ( q < static_cast<T>(0) ) ? p : ( static_cast<T>(1) - p );
    if ( r <= static_cast<T>(0) )
      quant = std::numeric_limits<T>::max();
    else {
      r = std::sqrt( -std::log(r) );
      if ( r <= Split_R ) {
	r -= Wichura_Constant_2;
	numer = Wichura_C.back();
	denom = Wichura_D.back();
	for ( nit = Wichura_C.rbegin() + 1, dit = Wichura_D.rbegin() + 1;
	      nit != Wichura_C.rend(); nit++, dit++ ) {
	  numer = numer * r + (*nit);
	  denom = denom * r + (*dit);
	}
	quant = numer / denom;
      }
      else {
	r -= Split_R;
	numer = Wichura_E.back();
	denom = Wichura_F.back();
	for ( nit = Wichura_E.rbegin() + 1, dit = Wichura_F.rbegin() + 1;
	      nit != Wichura_E.rend(); nit++, dit++ ) {
	  numer = numer * r + (*nit);
	  denom = denom * r + (*dit);
	}
	quant = numer / denom;
      }  // if (r <= Split_R) ... else ...
    }  // if (r <= 0.0) ... else ...
  }  // if (std::abs(q) <= Split_Q) ... else ...

  quant = std::abs( quant ) * ( (q >= static_cast<T>(0)) ? 1 : -1 );
  return quant;
};


