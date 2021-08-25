
#include <abseil/coordinate_systems/coordinate_system.hpp>


#ifndef _ABSEIL_SPHERICAL_COORDINATE_
#define _ABSEIL_SPHERICAL_COORDINATE_


namespace abseil {

  
  template< typename T = double >
  class spherical_coordinate :
    public coordinate_system<3, T>
  {
  public:
    typedef size_t size_type;
    typedef T value_type;

    explicit spherical_coordinate();
    explicit spherical_coordinate(
      const T rad = 1,
      const T inc = 1,
      const T azi = 1
    );

  
    template< typename InputIt >
    spherical_coordinate( InputIt first );

    spherical_coordinate<T>& operator=(
      const spherical_coordinate<T>& other
    );

    T central_angle( const spherical_coordinate<T>& other ) const;
    T distance( const spherical_coordinate<T>& other ) const;
    
    T radius() const;
    T inclination() const;
    T azimuth() const;

    void radius( const T r );
    void inclination( const T incl );
    void azimuth( const T azi );


  private:
    T central_angle_antipodal_(
      const spherical_coordinate<T>& other
    ) const;
  };



};
// namespace abseil


#include "abseil/coordinate_systems/spherical_coordinate_def.inl"


#endif  // _ABSEIL_SPHERICAL_COORDINATE_
