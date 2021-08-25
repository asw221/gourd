
#include <abseil/coordinate_systems/coordinate_system.hpp>


#ifndef _ABSEIL_CARTESIAN_COORDINATE_
#define _ABSEIL_CARTESIAN_COORDINATE_


namespace abseil {

  
  template< size_t Dim = 2, typename T = double >
  class cartesian_coordinate :
    public coordinate_system<Dim, T>
  {
  public:
    typedef size_t size_type;
    typedef T value_type;

    explicit cartesian_coordinate();
  
    template< typename InputIt >
    cartesian_coordinate( InputIt first );
  
    cartesian_coordinate<Dim, T>& operator=(
      const cartesian_coordinate<Dim, T>& other
    );

    T distance( const cartesian_coordinate<Dim, T>& other ) const;
    T taxicab( const cartesian_coordinate<Dim, T>& other ) const;
  };


};
// namespace abseil


#include "abseil/coordinate_systems/cartesian_coordinate_def.inl"


#endif  // _ABSEIL_CARTESIAN_COORDINATE_
