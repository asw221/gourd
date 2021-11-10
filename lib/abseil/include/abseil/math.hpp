
#include <complex>


#ifndef _ABSEIL_MATH_
#define _ABSEIL_MATH_



/*!
 * @namespace num
 * 
 * Templated numerical constants for pre c++20 compatibility.
 * Computed with WolframAlpha.
 *
 * Defined in file math.hpp
 */
namespace num {

  template< typename T > inline constexpr
  T e_v     = 2.718281828459045235360287471352662498;  /*!< e */

  template< typename T > inline constexpr
  T inv_pi_v = 0.318309886183790671537767526745028724; /*!< 1/pi */

  template< typename T > inline constexpr
  T inv_sqrt2_v   = 0.707106781186547524400844362104849039;  /*!< 1/sqrt(2) */
  
  template< typename T > inline constexpr
  T inv_sqrt2pi_v = 0.398942280401432677939946059934381868;  /*!< 1/sqrt(2 * pi) */

  template< typename T > inline constexpr
  T inv_sqrtpi_v  = 0.564189583547756286948079451560772586;  /*!< 1/sqrt(pi) */
    
  template< typename T > inline constexpr
  T ln2_v   = 0.693147180559945309417232121458176568;  /*!< log(2) */
    
  template< typename T > inline constexpr
  T pi_v    = 3.141592653589793238462643383279502884;  /*!< pi */

  template< typename T > inline constexpr
  T sqrt2_v = 1.414213562373095048801688724209698079;  /*!< sqrt(2) */
    
};




namespace abseil {


  /*! Signum function.
   * 
   * Defined in file math.h
   * @param val - Numeric scalar
   */
  template< typename T > inline
  int sgn( const T val ) {
    return static_cast<int>( (T(0) < val) - (val < T(0)) );
  };

  template< typename T > inline
  int sgn( const std::complex<T> val ) {
    const int s = sgn( val.real() );
    if ( s == 0 )  return sgn( val.imag() );
    return s;
  };


};


#endif  // _ABSEIL_MATH_
