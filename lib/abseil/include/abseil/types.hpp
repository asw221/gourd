
#include <type_traits>


#ifndef _ABSEIL_TYPES_
#define _ABSEIL_TYPES_

namespace abseil {

  template< typename... >
  using void_t = void;


  /*! Random access iterator checking
   */
  template< typename T, typename = void >
  struct is_random_iterator : std::false_type{;};

  template< typename T >
  struct is_random_iterator<T,
    void_t< decltype(++std::declval<T&>()),
	    decltype(--std::declval<T&>()),
	    decltype(*std::declval<T&>()),
	    decltype(std::declval<T&>() == std::declval<T&>())
    > >
    : std::true_type{;};


  template< typename T >
  using is_random_iterator_v = typename is_random_iterator<T>::value;
  
}  // namespace abseil

#endif  // _ABSEIL_TYPES_
