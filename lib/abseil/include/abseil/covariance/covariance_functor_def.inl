

#ifndef _ABSEIL_COVARIANCE_FUNCTOR_DEF_
#define _ABSEIL_COVARIANCE_FUNCTOR_DEF_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>  // std::distance

#include "abseil/covariance_functors.hpp"



// --- covariance_functor<T, M>::param_type -----------------------------

template< typename T, size_t M >
abseil::covariance_functor<T, M>::param_type::param_type(
  const param_type& other
) {
  assert( std::distance((&other)->_theta.cbegin(),
			(&other)->_theta.cend()) == M &&
	  "covariance_functor: parameter copy dim(LHS) != dim(RHS)" );
  std::copy( (&other)->_theta.cbegin(), (&other)->_theta.cend(),
	     this->_theta.data() );
};


template< typename T, size_t M >
template< typename InputIt >
abseil::covariance_functor<T, M>::param_type::param_type(
  InputIt first,
  InputIt last
) {
  assert( std::distance(first, last) == M &&
	  "covariance_functor: bad assignment pointer" ); 
  std::copy( first, last, _theta.data() );
};


// template< typename T, size_t M >
// abseil::covariance_functor<T, M>::param_type::param_type(
//   const int n
// ) {
//   _theta.resize( n, 1 );
// };



template< typename T, size_t M >
typename abseil::covariance_functor<T, M>::param_type &
abseil::covariance_functor<T, M>::param_type::operator= (
  const typename abseil::covariance_functor<T, M>::param_type& other
) {
  if ( this == &other )  return *this;
  std::copy( (&other)->_theta.cbegin(), (&other)->_theta.cend(),
	     this->_theta.data() );
  return *this;
};




template< typename T, size_t M >
T& abseil::covariance_functor<T, M>::param_type::operator[](
  const int pos
) {
  assert( pos >= 0 && pos < M &&
	  "covariance_functor: parameter index outside scope" );
  return _theta[pos];
};

template< typename T, size_t M >
const T& abseil::covariance_functor<T, M>::param_type::operator[](
  const int pos
) const {
  assert( pos >= 0 && pos < M &&
	  "covariance_functor: parameter index outside scope" );
  return _theta[pos];
};




// template< typename T >
// std::ostream& abseil::operator<<(
//   std::ostream& os,
//   const typename abseil::covariance_functor<T, M>::param_type& param
// ) {
//   os << "\u03B8" << " = (";
//   for ( unsigned i = 0; i < param.size(); i++ ) {
//     os << param[i] << ", ";
//   }
//   os << "\b\b) ";
//   return os;
// };



// --- covariance_functor<T, M> -----------------------------------------

template< typename T, size_t M >
abseil::covariance_functor<T, M>::covariance_functor(
  const covariance_functor<T, M>& other
) :
  _par(other._par)
{ ; }


// template< typename T, size_t M >
// abseil::covariance_functor<T, M>::covariance_functor(
//   const int n
// ) :
//   _par( n )
// { ; }


template< typename T, size_t M >
T abseil::covariance_functor<T, M>::operator() ( const T val ) const {
  return val;
};

template< typename T, size_t M >
T abseil::covariance_functor<T, M>::inverse( const T cov ) const {
  return cov;
};


template< typename T, size_t M >
T abseil::covariance_functor<T, M>::fwhm() const {
  return static_cast<T>( HUGE_VAL );
};


template< typename T, size_t M >
std::array<T, M>
abseil::covariance_functor<T, M>::gradient( const T val ) const {
  std::array<T, M> out;
  for ( size_t i = 0; i < M; i++ )  out[i] = 0;
  return out;
};


template< typename T, size_t M >
std::array<T, M>
abseil::covariance_functor<T, M>::param_lower_bounds() const {
  std::array<T, M> bound;
  for ( size_t i = 0; i < M; i++ )  bound[i] = 0;
  return bound;
};


template< typename T, size_t M >
std::array<T, M>
abseil::covariance_functor<T, M>::param_upper_bounds() const {
  std::array<T, M> bound;
  for ( size_t i = 0; i < M; i++ )
    bound[i] = static_cast<T>(HUGE_VAL);
  return bound;
};



template< typename T, size_t M >
void abseil::covariance_functor<T, M>::param(
  const abseil::covariance_functor<T, M>::param_type& par
) {
  _par = par;
};


template< typename T, size_t M >
typename abseil::covariance_functor<T, M>::param_type
abseil::covariance_functor<T, M>::param() const {
  return _par;
};


#endif  // _ABSEIL_COVARIANCE_FUNCTOR_DEF_
