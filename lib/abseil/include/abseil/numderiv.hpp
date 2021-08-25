
#include <cassert>


#ifndef _ABSEIL_NUMDERIV_
#define _ABSEIL_NUMDERIV_


namespace abseil {
  

  /*! Numerical gradient
   * 
   * Compute the numerical gradient of a function/functor with
   * respect to input \c x. 
   *
   * \c grad will compute a three point (central) finite difference
   * difference approximation of the gradient.
   */
  template< typename FuncT, typename T = typename FuncT::result_type >
  T grad( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x + h) - f(x - h)) / (2 * h);
  };

  template< typename FuncT, typename T = typename FuncT::result_type >
  T gradf( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x + h) - f(x)) / h;
  };

  template< typename FuncT, typename T = typename FuncT::result_type >
  T gradb( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x) - f(x - h)) / h;
  };


};


#endif  // _ABSEIL_NUMDERIV_
