
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "abseil/math.hpp"

#ifndef _ABSEIL_BESSEL_
#define _ABSEIL_BESSEL_


namespace abseil {
namespace bessel {


  template< typename T >
  void temme_ik(
    const T nu, const T x,
    T& rkmu, T& rk1,
    const int maxit = 1e4,
    const T tol = std::numeric_limits<T>::epsilon()
  ) {
    assert( abs(x) <= 2 && "temme_ik: x too large" );
    assert( abs(nu) <= 0.5 && "temme_ik: nu too large" );
    /* Rapid convergence for |x| <= 2 */
    //
    const T gp = std::tgamma(1 + nu);
    const T gm = std::tgamma(1 - nu);
    const T a = log(x / 2);
    const T b = exp(nu * a);
    const T sigma = -a * nu;
    const T c = (std::abs(nu) < tol) ? T(1) :
      std::sin(nu * num::pi_v<T>) / (nu * num::pi_v<T>);
    const T d = std::abs(sigma) < tol ? T(1) :
      std::sinh(sigma) / sigma;
    const T gam1 = (std::abs(nu) < tol) ? -num::e_v<T> :
      (0.5 / nu) * (gp - gm) * c;
    const T gam2 = (gp + gm) * c / 2;

    // Initial values
    T p = gp / (2 * b);
    T q = gm * b / 2;
    T f = (std::cosh(sigma) * gam1 + d * (-a) * gam2) / c;
    T h = p;
    T coef = 1;
    T sum = coef * f;
    T sum1 = coef * h;
    int k;  // Series variable
    for ( k = 1; k <= maxit; k++ ) {
      f = (k * f + p + q) / (k*k - nu*nu);
      p /= k - nu;
      q /= k + nu;
      h = p - k * f;
      coef *= x * x / (4 * k);
      sum += coef * f;
      sum1 += coef * h;
      if ( std::abs(coef * f) < abs(sum) * tol ) break;
    }
    if ( k > maxit )
      std::cerr << "\t*** WARNING: temme_ik: no convergence\n";
    //
    rkmu = sum;
    rk1 = 2 * sum1 / x;
  };



  template< typename T >
  void cf2(
    const T nu, const T x,
    T& rkmu, T& rk1,
    const int maxit = 1e4,
    const T tol = std::numeric_limits<T>::epsilon()
  ) {
    assert( std::abs(x) > 1 && "bessel::cf2: |x| <= 1" );
    // |x| >= |nu|, continued fraction converges rapidly
    // |x| -> 0, continued fraction fails to converge

    // Calculate K(nu, x) and K(nu+1, x) by evaluating continued
    // fraction
    // z1 / z0 = U(nu + 1.5, 2nu + 1, 2x) / U(nu + 0.5, 2nu + 1, 2x)
    
    /* Steed's algorithm: see S 5.2 in "Numerical Recipes in C".
     *
     * Also:
     * Thompson and Barnett, J Comp Phys, vol 64, 490 (1986)
     * Thompson and Barnett, Comp Phys Commun, vol 47, 245 (1987)
     */
    // Initial values
    T a = nu * nu - 0.25;
    T b = 2 * (x + 1);      // b1
    T d = 1 / b;            // d1 = 1 / b1
    T f = d;                
    T delta = d;
    T prev = 0;             // q0
    T current = 1;          // q1
    T c = -a;
    T q = c;                // q1 = c1
    T s = 1 + q * delta;    // s1
    int k;                  // Series variable
    for (k = 2; k < maxit; k++) {
      // series summation s = 1 + \sum_{n=1}^{\infty} c_n * z_n / z_0
      // continued fraction f = z1 / z0
      a -= 2 * (k - 1);
      b += 2;
      d = 1 / (b + a * d);
      delta *= b * d - 1;
      f += delta;
      //
      q = (prev - (b - 2) * current) / a;
      prev = current;
      current = q;                        // forward recurrence for q
      c *= -a / k;
      q += c * q;
      s += q * delta;
      //
      if ( q < tol ) {
	c *= q;
	prev /= q;
	current /= q;
	q = 1;
      }
      if ( std::abs(q * delta) < std::abs(s) * tol ) break;
    }
    if ( k > maxit )
      std::cerr << "\t*** WARNING: cf series did not converge\n";
    //
    if ( x >= (std::numeric_limits<T>::max_exponent - 1) )
      rkmu = std::exp( 0.5 * std::log(num::pi_v<T> / (2 * x)) -
		       x - std::log(s) );
    else
      rkmu = std::sqrt(num::pi_v<T> / (2 * x)) * std::exp(-x) / s;
    rk1 = rkmu * (0.5 + nu + x + (nu * nu - 0.25) * f) / x;
  };


  
  template< typename T >
  void bess_nni(
    const T x, const T nu, const int nrecur,
    T& i, T& i_prime, T& i_1, T& i_prime_1,
    const int maxit = 1e4,
    const T tol = std::numeric_limits<T>::epsilon()
  ) {
    const T fpmin = std::numeric_limits<T>::min();
    const T xinv = 1 / x;
    T b = 2 * xinv * nu;
    T h = nu / x;
    h = (h < fpmin) ? fpmin : h;
    T c = h, d = 0, delta;
    int k;
    for ( k = 0; k < maxit; k++ ) {
      b += 2 * xinv;
      d = 1 / (b + d);
      c = b + 1 / c;
      delta = c * d;
      h = delta * h;
      if ( std::abs(delta - 1) < tol ) break;
    }
    if ( k == (maxit - 1) )
      std::cerr << "\t*** WARNING: bess_i: maxit reached\n";
    //
    i = fpmin;
    i_prime = h * fpmin;
    i_1 = i;
    i_prime_1 = i_prime;
    T fact = nu / x;
    for ( int j = nrecur; j >= 1; j-- ) {
      T tmp = fact * i + i_prime;
      fact -= xinv;
      i_prime = fact * tmp + i;
      i = tmp;
    }
    /* Now have unnormalized I */  
  };


  



  template< typename T >
  void bessik(
    const T x, const T nu,
    T& ri, T& rk, T& rip, T& rkp, /* Return values I, K, I', K' */
    const int maxit = 1e4,
    const T tol = std::numeric_limits<T>::epsilon()
  ) {
    assert( x > 0 && "bessik: x must be > 0" );
    assert( nu >= 0 && "bessik: nu must be >= 0" );  // <- Bad
    namespace bessy = abseil::bessel;
    const int nrecur = static_cast<int>(nu + 0.5);
    /* ^^ Number of downward recurrences of I's/ upward recurrences
     * of the K's */
    const T xmu = nu - nrecur;
    const T xinv = 1 / x;
    T k_mu, k_1;
    T i, i_prime, i_1, i_prime_1;
    /* Compute non-normalized I, I' */
    bessy::bess_nni(x, nu, nrecur,
		    i, i_prime, i_1, i_prime_1, maxit, tol);
    //
    if ( x < 2 ) {  /* Use Temme series */
      bessy::temme_ik(xmu, x, k_mu, k_1, maxit, tol);
    }
    else {  /* Use continued fraction series */
      bessy::cf2(xmu, x, k_mu, k_1, maxit, tol);
    }
    //
    const T k_mu_p = xmu * xinv * k_mu - k_1;
    const T i_mu = xinv / (i_prime / i * k_mu - k_mu_p);
    ri  = i_mu * i_1 / i;
    rip = i_mu * i_prime_1 / i;
    for ( int i = 1; i <= nrecur; i++ ) {
      T tmp = (xmu + i) * 2 * xinv * k_1 + k_mu;
      k_mu = k_1;  k_1 = tmp;
    }
    rk = k_mu;
    rkp = nu * xinv * k_mu - k_1;
  };

  
}  // namespace bessel

  

  /*! Bessel K function
   *
   * Also called modified or cylindrical bessel function of the 
   * second kind.
   *
   * Less accurate than Boost's implementation for |x| < 2 and nu -> 0
   */
  template< typename T >
  T cyl_bessel_k( const T nu, const T x ) {
    T i, k, deriv_i, deriv_k;
    abseil::bessel::bessik( x, nu, i, k, deriv_i, deriv_k );
    return k;
  };
  
}  // namespace abseil


#endif  // _ABSEIL_BESSEL_






