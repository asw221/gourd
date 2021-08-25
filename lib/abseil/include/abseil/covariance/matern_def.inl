
#ifndef _ABSEIL_MATERN_DEF_
#define _ABSEIL_MATERN_DEF_


#include <array>
#include <cassert>
#include <cmath>
#include <iterator>  // std::distance
#include <limits>
#include <stdexcept>

#include <boost/math/special_functions/bessel.hpp>

#include "abseil/math.hpp"
#include "abseil/numderiv.hpp"



// --- matern<T> -----------------------------------------------------


template< typename T >
T abseil::matern<T>::_eps = 1e-6;

template< typename T >
T abseil::matern<T>::_tol = 1e-6;

template< typename T >
int abseil::matern<T>::_max_it = 100;



template< typename T >
abseil::matern<T>::matern() :
  covariance_functor<T, 3>()
{
  _compute_normalizing_constant();
};



template< typename T >
abseil::matern<T>::matern(
  const abseil::matern<T>::param_type & par
) :
  covariance_functor<T, 3>( par )
{
  _compute_normalizing_constant();
};


template< typename T >
template< typename InputIt >
abseil::matern<T>::matern( InputIt first, InputIt last ) :
  covariance_functor<T, 3>(first, last)
{
  _compute_normalizing_constant();
};




template< typename T >
void abseil::matern<T>::_compute_normalizing_constant() {
  _sqrt_2nu_rho = std::sqrt( 2 * nu() ) / rho();
  // _norm_c = variance() * std::pow( static_cast<T>(2), 1 - nu()) *
  //   std::pow( _sqrt_2nu_rho, nu() ) /
  //   std::tgamma( nu() );
  _norm_c = std::log( variance() ) +
    ( 1 - nu() ) * num::ln2_v<T> +
    nu() * std::log( _sqrt_2nu_rho ) -
    std::lgamma( nu() );
  //
  _norm_c = std::exp( _norm_c );
};



template< typename T >
T abseil::matern<T>::operator() ( const T val ) const {
  if ( val == T(0) ) {
    return this->_par[0];
  }
  return _norm_c *
    std::pow( std::abs(val), nu() ) *
    boost::math::cyl_bessel_k( nu(), _sqrt_2nu_rho * std::abs(val) );
};



template< typename T >
T abseil::matern<T>::inverse( const T val ) const {
  T x = rho();
  T diff = std::numeric_limits<T>::infinity();
  int iter = 0;
  while ( std::abs(diff) > _tol  &&  iter < _max_it ) {
    diff = ( this->operator()(x) - val ) /
      abseil::gradf( *this, x );
    x -= diff;
    iter++;
  }
  if ( iter >= _max_it && diff > _tol ) {
    std::cerr << "  ** matern::inverse : did not converge\n";
  }
  return x;
};


template< typename T >
T abseil::matern<T>::fwhm() const {
  return 2 * std::abs( inverse( variance() / 2 ) );
};



template< typename T >
std::array<T, 3>
abseil::matern<T>::gradient( const T val ) const {
  const T c = operator()( val );
  param_type theta = this->_par;
  matern<T> cov_tilde;
  std::array<T, 3> gr;
  gr[0] = c / variance();
  //
  theta[1] += _eps;
  cov_tilde.param( theta );
  gr[1] = (cov_tilde(val) - c) / _eps;
  //
  theta[1] -= _eps;
  theta[2] += _eps;
  cov_tilde.param( theta );
  gr[2] = (cov_tilde(val) - c) / _eps;
  //
  return gr;
};




template< typename T >
std::array<T, 3>
abseil::matern<T>::param_lower_bounds() const {
  std::array<T, 3> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::array<T, 3>
abseil::matern<T>::param_upper_bounds() const {
  const T huge = static_cast<T>( HUGE_VAL );
  /* Bessel computation can lead to overflow errors if the order
   * nu is too large. While I have not studies this rigorously,
   * values of nu > 32.5 (approx) lead to overflow errors for 
   * x <= 1e-8
   */
  std::array<T, 3> bound{ huge, huge, 32.5 };
  return bound;
};




template< typename T >
T abseil::matern<T>::variance() const {
  return this->_par[0];
};

template< typename T >
T abseil::matern<T>::nu() const {
  return this->_par[2];
};


template< typename T >
T abseil::matern<T>::rho() const {
  return this->_par[1];
};



template< typename T >
void abseil::matern<T>::param(
  const abseil::matern<T>::param_type& par
) {
  assert( par[0] > 0 && "Invalid parameter (0)" );
  assert( par[1] > 0 && "Invalid parameter (1)" );
  assert( par[2] > 0 && "Invalid parameter (2)" );
  this->_par = par;
  _compute_normalizing_constant();
};



template< typename T >
void abseil::matern<T>::variance( const T val ) {
  assert( val > 0 && "Invalid variance parameter" );
  this->_par[0] = val;
  _compute_normalizing_constant();
};


template< typename T >
void abseil::matern<T>::nu( const T val ) {
  assert( val > 0 && "Invalid order parameter" );
  this->_par[2] = val;
  _compute_normalizing_constant();
};


template< typename T >
void abseil::matern<T>::rho( const T val ) {
  assert( val > 0 && "Invalid inverse bandwidth parameter" );
  this->_par[1] = val;
  _compute_normalizing_constant();
};


#endif  // _ABSEIL_MATERN_DEF_
