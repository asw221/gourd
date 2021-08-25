

#include <abseil/coordinate_systems/cartesian_coordinate.hpp>
#include <abseil/coordinate_systems/spherical_coordinate.hpp>
#include <algorithm>
#include <cmath>


template< typename T >
abseil::spherical_coordinate<T> abseil::to_spherical(
  const abseil::cartesian_coordinate<3, T>& xyz
) {
  const T eps0 = 1e-6;
  const T x = (std::abs( xyz[0] ) < eps0) ?
    ( sgn(xyz[0]) == 0 ? eps0 : sgn(xyz[0]) * eps0 ) :
    xyz[0];
  const T r = std::sqrt( xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2] );
  const T inc = std::acos( xyz[2] / std::max( r, eps0 ) );
  const T azi = std::atan2( xyz[1], x );
  return abseil::spherical_coordinate<T>( r, inc, azi );
};


template< typename T >
abseil::cartesian_coordinate<3, T> abseil::to_cartesian(
  const abseil::spherical_coordinate<T>& coord
) {
  const T sin_incl = std::sin( coord.inclination() );
  const T r = coord.radius();
  abseil::cartesian_coordinate<3, T> xyz;
  xyz[0] = r * sin_incl * std::cos( coord.azimuth() );
  xyz[1] = r * sin_incl * std::sin( coord.azimuth() );
  xyz[2] = r * std::cos( coord.inclination() );
  return xyz;
};
