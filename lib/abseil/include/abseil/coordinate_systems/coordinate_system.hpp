
#include <abseil/math.hpp>
#include <algorithm>
#include <array>
#include <cassert>


#ifndef _ABSEIL_COORDINATE_SYSTEM_
#define _ABSEIL_COORDINATE_SYSTEM_


namespace abseil {

  
  template< size_t Dim = 2, typename T = double >
  class coordinate_system {
  public:
    typedef size_t size_type;
    typedef T value_type;

    explicit coordinate_system();

    template< typename InputIt >
    explicit coordinate_system( InputIt first );

    coordinate_system<Dim, T>& operator=(
      const coordinate_system<Dim, T>& other
    );
    
    constexpr size_type dim() const { return Dim; };
    
    T& operator[]( const size_type pos );
    const T& operator[]( const size_type pos ) const;

    bool operator==( const coordinate_system<Dim, T>& other ) const;
    bool operator!=( const coordinate_system<Dim, T>& other ) const;

    T distance( const coordinate_system<Dim, T>& other ) const;


    friend std::ostream& operator<<(
      std::ostream& os,
      const coordinate_system<Dim, T>& coord
    ) {
      os << " (";
      for ( const value_type& val : coord.coord_ )  os << val << ", ";
      os << "\b\b) ";
      return os;
    };
  
  protected:
    std::array<T, Dim> coord_;
  };




};
// namespace abseil



template< size_t Dim, typename T >
abseil::coordinate_system<Dim, T>::coordinate_system()  { ; }

//   coord_(Dim) { ; }



template< size_t Dim, typename T >
template< typename InputIt >
abseil::coordinate_system<Dim, T>::coordinate_system( InputIt first ) {
  assert( &(*first) + Dim &&
	  "coordinage_system: invalid iterator constructor" );
  std::copy( first, first + Dim, coord_.data() );
};
  
//  coord_( first, first + Dim ) { ; }



template< size_t Dim, typename T >
T& abseil::coordinate_system<Dim, T>::operator[](
  const abseil::coordinate_system<Dim, T>::size_type pos
) {
  return coord_[pos];
};


template< size_t Dim, typename T >
const T& abseil::coordinate_system<Dim, T>::operator[](
  const abseil::coordinate_system<Dim, T>::size_type pos
) const {
  return coord_[pos];
};


template< size_t Dim, typename T >
abseil::coordinate_system<Dim, T>&
abseil::coordinate_system<Dim, T>::operator=(
  const abseil::coordinate_system<Dim, T>& other
) {
  if ( this == &other )  return *this;
  // coord_.assign( other.coord_.cbegin(), other.coord_.cend() );
  std::copy( (&other)->coord_.cbegin(), (&other)->coord_.cend(),
	     this->coord_.data() );
  return *this;
};


template< size_t Dim, typename T >
bool abseil::coordinate_system<Dim, T>::operator==(
  const abseil::coordinate_system<Dim, T>& other
) const {
  return this->coord_ == other.coord_;
};


template< size_t Dim, typename T >
bool abseil::coordinate_system<Dim, T>::operator!=(
  const abseil::coordinate_system<Dim, T>& other
) const {
  return !this->operator==( other );
};


template< size_t Dim, typename T >
T abseil::coordinate_system<Dim, T>::distance(
  const abseil::coordinate_system<Dim, T>& other
) const {
  T d = 0;
  for ( size_type i = 0; i < Dim; i++ ) {
    d += std::abs( coord_[i] - other.coord_[i] );
  }
  return d;
};




#endif  // _ABSEIL_COORDINATE_SYSTEM_
