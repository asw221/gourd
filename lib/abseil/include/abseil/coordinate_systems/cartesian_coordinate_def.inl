
#include <algorithm>


template< size_t Dim, typename T >
abseil::cartesian_coordinate<Dim, T>::cartesian_coordinate() :
  abseil::coordinate_system<Dim, T>() { ; }


template< size_t Dim, typename T >
template< typename InputIt >
abseil::cartesian_coordinate<Dim, T>::cartesian_coordinate( InputIt first ) :
  abseil::coordinate_system<Dim, T>( first ) { ; }



template< size_t Dim, typename T >
abseil::cartesian_coordinate<Dim, T>&
abseil::cartesian_coordinate<Dim, T>::operator=(
  const abseil::cartesian_coordinate<Dim, T>& other
) {
  if ( this == &other )  return *this;
  // this->coord_.assign( (&other)->coord_.cbegin(), (&other)->coord_.cend() );
  std::copy( (&other)->coord_.cbegin(), (&other)->coord_.cend(),
	     this->coord_.data() );
  return *this;
};


template< size_t Dim, typename T >
T abseil::cartesian_coordinate<Dim, T>::distance(
  const abseil::cartesian_coordinate<Dim, T>& other
) const {
  T d = 0, temp;
  for ( size_type i = 0; i < Dim; i++ ) {
    temp = this->coord_[i] - other[i];
    d += temp * temp;
  }
  d = std::sqrt( d );
  return d;
};



template< size_t Dim, typename T >
T abseil::cartesian_coordinate<Dim, T>::taxicab(
  const abseil::cartesian_coordinate<Dim, T>& other
) const {
  T d = 0;
  for ( size_type i = 0; i < Dim; i++ ) {
    d += std::abs( this->coord_[i] - other[i] );
  }
  return d;
};

