
#ifndef _ABSEIL_RATIONAL_QUADRATIC_DEF_
#define _ABSEIL_RATIONAL_QUADRATIC_DEF_


#include <array>
#include <cassert>
#include <cmath>
#include <iterator>  // std::distance
#include <limits>
#include <stdexcept>



// --- rational_quadratic<T> -----------------------------------------


template< typename T >
abseil::rational_quadratic<T>::rational_quadratic() :
  covariance_functor<T, 3>()
{
  _validate_parameters();
};



template< typename T >
abseil::rational_quadratic<T>::rational_quadratic(
  const abseil::rational_quadratic<T>::param_type & par
) :
  covariance_functor<T, 3>( par )
{
  _validate_parameters();
};


template< typename T >
template< typename InputIt >
abseil::rational_quadratic<T>::rational_quadratic(
  InputIt first,
  InputIt last
) :
  covariance_functor<T, 3>(first, last)
{
  _validate_parameters();
};




template< typename T >
T abseil::rational_quadratic<T>::operator() ( const T val ) const {
  return variance() * std::pow(1 + psi() / nu() * val * val, -nu());
};



template< typename T >
T abseil::rational_quadratic<T>::inverse( const T val ) const {
  const T a = std::pow(variance()/val, 1/nu()) - 1;
  return std::sqrt( nu() * a / psi() );
};


template< typename T >
T abseil::rational_quadratic<T>::fwhm() const {
  return 2 * std::abs( inverse( variance() / 2 ) );
};



template< typename T >
std::array<T, 3>
abseil::rational_quadratic<T>::gradient( const T val ) const {
  const T valsq = val * val;
  const T a = 1 + valsq * psi() / nu();
  const T b = std::pow(a, -nu());
  std::array<T, 3> gr;
  gr[0] = b;
  gr[1] = -variance() * valsq * b / a;
  gr[2] = variance() * b * ((a - 1)/a - std::log(a));
  return gr;
};




template< typename T >
std::array<T, 3>
abseil::rational_quadratic<T>::param_lower_bounds() const {
  std::array<T, 3> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::array<T, 3>
abseil::rational_quadratic<T>::param_upper_bounds() const {
  const T huge = static_cast<T>( HUGE_VAL );
  std::array<T, 3> bound{ huge, huge, huge };
  return bound;
};




template< typename T >
T abseil::rational_quadratic<T>::variance() const {
  return this->_par[0];
};

template< typename T >
T abseil::rational_quadratic<T>::nu() const {
  return this->_par[2];
};


template< typename T >
T abseil::rational_quadratic<T>::psi() const {
  return this->_par[1];
};



template< typename T >
void abseil::rational_quadratic<T>::param(
  const abseil::rational_quadratic<T>::param_type& par
) {
  this->_par = par;
  _validate_parameters();
};



template< typename T >
void abseil::rational_quadratic<T>::variance( const T val ) {
  assert( val > 0 && "Invalid variance parameter" );
  this->_par[0] = val;
};


template< typename T >
void abseil::rational_quadratic<T>::nu( const T val ) {
  assert( val > 0 && "Invalid order parameter" );
  this->_par[2] = val;
};


template< typename T >
void abseil::rational_quadratic<T>::psi( const T val ) {
  assert( val > 0 && "Invalid inverse bandwidth parameter" );
  this->_par[1] = val;
};



template< typename T >
void abseil::rational_quadratic<T>::_validate_parameters() const {
  if ( this->_par.size() != 3 ) {
    throw std::domain_error(
      "rational_quadratic functor: parameter must be size = 3" );
  }
  if ( this->_par[0] <= 0 ) {
    throw std::domain_error(
      "rational_quadratic functor: variance parameter"
      " must be > 0" );
  }
  if ( this->_par[1] <= 0 ) {
    throw std::domain_error(
      "rational_quadratic functor: bandwidth parameter"
      " must be > 0" );
  }
  if ( this->_par[2] <= 0 ) {
    throw std::domain_error(
      "rational_quadratic functor: exponent parameter"
      " must be > 0" );
  }
};

#endif  // _ABSEIL_RATIONAL_QUADRATIC_DEF_
