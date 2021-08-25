
#include <abseil/coordinate_systems/coordinate_system.hpp>
#include <abseil/coordinate_systems/cartesian_coordinate.hpp>
#include <abseil/coordinate_systems/spherical_coordinate.hpp>



#ifndef _ABSEIL_COORDINATES_
#define _ABSEIL_COORDINATES_


namespace abseil {


  /*!
   * Cartesian to spherical coordinate conversion
   * @param xyz - 3D cartesian coordinate
   */
  template< typename T >
  abseil::spherical_coordinate<T> to_spherical(
    const abseil::cartesian_coordinate<3, T>& xyz
  );


  /*!
   * Spherical to cartesian coordinate conversion
   * @param coord - Spherical coordinate
   */
  template< typename T >
  abseil::cartesian_coordinate<3, T> to_cartesian(
    const abseil::spherical_coordinate<T>& coord
  );


  

};
// namespace abseil


#include "abseil/coordinate_systems/coordinates_def.inl"


#endif  // _ABSEIL_COORDINATES_
