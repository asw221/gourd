
#include <cassert>
#include <iterator>
#include <type_traits>

#include "abseil/types.hpp"


#ifndef _ABSEIL_CHEBYSHEV_
#define _ABSEIL_CHEBYSHEV_


namespace abseil {


  /*! Chebyshev evaluation
   *
   * The Chebyshev polynomial,
   *   \sum_{k=0}^{m-1} c_k T_k(y) - c_0/2,
   * evaluated at a point,
   *   y = [x - (b + a)/2] / [(b - a)/2].
   *
   * All arguments are input.
   * Modified from "Numerical Recipes in C".
   *
   * @return y
   */
  template< typename T = double, typename IterT >
  T chebev(
    const T a, const T b,               /*!< Interval anchors */
    const IterT cbeg, const IterT cend, /*!< Chebyshev coeffs */
    const T x                           /*!< Evaluation center */
  ) {
    static_assert(std::is_floating_point<T>::value,
		  "chebev: inputs must be a floating point type");
    static_assert(abseil::is_random_iterator<IterT>::value,
		  "chebev: template type not iterable");
    assert( (x-a) * (x-b) <= (T)0 && "chebev: x out of range" );
    assert( cbeg <= cend && "chebev: unordered coefficients" );
    const T y = (2 * x - a - b) / (b - a);
    const T y2 = 2 * y;
    T d = 0, dd = 0, sv;
    for ( IterT coefit = cend; coefit != cbeg; --coefit ) {
      sv = d;
      d  = y2 * d - dd + (*coefit);
      dd = sv;
    }
    return y * d - dd + 0.5 * (*cbeg);
  };
  
}  // namespace abseil


#endif  // _ABSEIL_CHEBYSHEV_
