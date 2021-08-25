

#ifndef _ABSEIL_RADIAL_BASIS_DEF_
#define _ABSEIL_RADIAL_BASIS_DEF_


#include <array>
#include <cassert>
#include <cmath>
#include <iterator>  // std::distance
#include <stdexcept>

#include "abseil/covariance_functors.hpp"


// --- radial_basis<T> -----------------------------------------------

template< typename T >
template< typename InputIt >
abseil::radial_basis<T>::radial_basis(
  InputIt first,
  InputIt last
) :
  covariance_functor<T, 3>(first, last)
{
  _validate_parameters();
};



template< typename T >
abseil::radial_basis<T>::radial_basis(
  const abseil::radial_basis<T>::param_type & par
) :
  covariance_functor<T, 3>( par )
{
  _validate_parameters();
};




template< typename T >
T abseil::radial_basis<T>::operator() ( const T val ) const {
  return variance() *
    std::exp( -bandwidth() *
	      std::pow(std::abs(val), exponent()) );
};


template< typename T >
T abseil::radial_basis<T>::inverse( const T cov ) const {
  assert( cov > 0 && "radial_basis: argument to inverse must be > 0");
  return std::pow( -std::log(cov / variance()) / bandwidth(),
		   1 / exponent() );
};


template< typename T >
T abseil::radial_basis<T>::fwhm() const {
  return 2.0 * std::pow( std::log(2.0) / bandwidth(), 1 / exponent());
};



template< typename T >
std::array<T, 3>
abseil::radial_basis<T>::gradient( const T val ) const {
  const T c = operator()( val );
  std::array<T, 3> gr;
  gr[0] = c / variance();
  gr[1] = -std::pow( std::abs(val), exponent() ) * c;
  gr[2] = gr[1] * bandwidth() * std::log( std::abs(val) + 1e-8 );
  return gr;
};



template< typename T >
std::array<T, 3>
abseil::radial_basis<T>::param_lower_bounds() const {
  std::array<T, 3> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::array<T, 3>
abseil::radial_basis<T>::param_upper_bounds() const {
  const T huge = static_cast<T>( HUGE_VAL );
  std::array<T, 3> bound{ huge, huge, 2.0 };
  return bound;
};


// f(d) := tau^2 * exp( -psi |d|^nu )
// d/d(psi) f(d) = -|d|^nu * f(d)
// d/d(nu) f(d) = -psi * |d|^nu * log|d| * f(d)



template< typename T >
T abseil::radial_basis<T>::variance() const {
  return this->_par[0];
};

template< typename T >
T abseil::radial_basis<T>::bandwidth() const {
  return this->_par[1];
};

template< typename T >
T abseil::radial_basis<T>::exponent() const {
  return this->_par[2];
};



template< typename T >
void abseil::radial_basis<T>::param(
  const abseil::radial_basis<T>::param_type& par
) {
  assert( par[0] > 0 && "Invalid parameter (0)");
  assert( par[1] > 0 && "Invalid parameter (1)");
  assert( par[2] > 0 && par[2] <= 2 && "Invalid parameter (2)");
  this->_par = par;
};


template< typename T >
void abseil::radial_basis<T>::variance( const T val ) {
  if ( val <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: variance parameter"
      " must be > 0" );
  }
  this->_par[0] = val;
};

template< typename T >
void abseil::radial_basis<T>::bandwidth( const T val ) {
  if ( val <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: bandwidth parameter"
      " must be > 0" );
  }
  this->_par[1] = val;
};

template< typename T >
void abseil::radial_basis<T>::exponent( const T val ) {
  if ( val <= 0 || val > 2 ) {
    throw std::domain_error(
      "radial_basis functor: exponent parameter"
      " must be on (0, 2]" );
  }
  this->_par[2] = val;
};


template< typename T >
void abseil::radial_basis<T>::_validate_parameters() const {
  if ( this->_par.size() != 3 ) {
    throw std::domain_error(
      "radial_basis functor: parameter must be size = 3" );
  }
  if ( this->_par[0] <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: variance parameter"
      " must be > 0" );
  }
  if ( this->_par[1] <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: bandwidth parameter"
      " must be > 0" );
  }
  if ( this->_par[2] <= 0 || this->_par[2] > 2 ) {
    throw std::domain_error(
      "radial_basis functor: exponent parameter"
      " must be on (0, 2]" );
  }
};


#endif  // _ABSEIL_RADIAL_BASIS_DEF_

