
#include <abseil/math.hpp>
#include <algorithm>
#include <cmath>



template< typename T >
abseil::spherical_coordinate<T>::spherical_coordinate() :
  abseil::coordinate_system<3, T>() { ; }


template< typename T >
abseil::spherical_coordinate<T>::spherical_coordinate(
  const T rad,
  const T inc,
  const T azi
) : abseil::coordinate_system<3, T>() {
  this->coord_[0] = std::abs( rad );
  this->coord_[1] = inc;
  this->coord_[2] = azi;
};


template< typename T >
template< typename InputIt >
abseil::spherical_coordinate<T>::spherical_coordinate( InputIt first ) :
  abseil::coordinate_system<3, T>( first ) { ; }



template< typename T >
abseil::spherical_coordinate<T>&
abseil::spherical_coordinate<T>::operator=(
  const abseil::spherical_coordinate<T>& other
) {
  if ( this == &other )  return *this;
  // this->coord_.assign( (&other)->coord_.cbegin(), (&other)->coord_.cend() );
  std::copy( (&other)->coord_.cbegin(), (&other)->coord_.cend(),
	     this->coord_.data() );
  return *this;
};


template< typename T >
T abseil::spherical_coordinate<T>::central_angle(
  const abseil::spherical_coordinate<T>& other
) const {
  const T eps0 = 1e-5;
  T sin_dincl_2, sin_dazi_2, angle;
  sin_dincl_2 =
    std::sin( std::abs( inclination() - other.inclination() ) / 2 );
  sin_dazi_2 =
    std::sin( std::abs( azimuth() - other.azimuth() ) / 2 );
  if ( std::abs(sin_dazi_2) + eps0 > 1  &&
       std::abs(sin_dincl_2) < eps0
       ) {
    // Use antipodal formula
    return central_angle_antipodal_( other );
  }
  angle = sin_dazi_2 * sin_dazi_2 +
    std::cos( azimuth() ) * std::cos( other.azimuth() ) *
    sin_dincl_2 * sin_dincl_2;
  angle = 2 * std::asin( std::sqrt(angle) );
  return angle;
};



template< typename T >
T abseil::spherical_coordinate<T>::distance(
  const abseil::spherical_coordinate<T>& other
) const {
  assert( std::abs( radius() - other.radius() ) < 1e-4 &&
	  "spherical_coordinate::distance unequal radii" );
  const T cangle = central_angle( other );
  return radius() * cangle;
};



template< typename T >
T abseil::spherical_coordinate<T>::central_angle_antipodal_(
  const abseil::spherical_coordinate<T>& other
) const {
  const T eps0 = 1e-6;
  const T cos_azi = std::cos( azimuth() );
  const T cos_azi_oth = std::cos( other.azimuth() );
  const T sin_azi = std::sin( azimuth() );
  const T sin_azi_oth = std::sin( other.azimuth() );
  const T cos_dincl = std::cos( std::abs(inclination() - other.inclination()) );
  const T sin_dincl = std::sin( std::abs(inclination() - other.inclination()) );
  const T A = cos_azi_oth * sin_dincl;
  const T B = cos_azi * sin_azi_oth -
    sin_azi * cos_azi_oth * cos_dincl;
  T C = sin_azi * sin_azi_oth +
    cos_azi * cos_azi_oth * cos_dincl;
  C = (std::abs(C) < eps0) ? abseil::sgn(C) * eps0 : C;
  C = (C == 0) ? eps0 : C;
  return std::atan2( std::sqrt(A * A + B * B), C );
};






template< typename T >
T abseil::spherical_coordinate<T>::radius() const {
  return this->coord_[0];
};


template< typename T >
T abseil::spherical_coordinate<T>::inclination() const {
  return this->coord_[1];
};


template< typename T >
T abseil::spherical_coordinate<T>::azimuth() const {
  return this->coord_[2];
};




template< typename T >
void abseil::spherical_coordinate<T>::radius( const T r ) {
  this->coord_[0] = std::abs( r );
};


template< typename T >
void abseil::spherical_coordinate<T>::inclination( const T incl ) {
  this->coord_[1] = incl;
};


template< typename T >
void abseil::spherical_coordinate<T>::azimuth( const T azi ) {
  this->coord_[2] = azi;
};


