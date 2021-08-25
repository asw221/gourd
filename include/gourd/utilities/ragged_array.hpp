
#include <vector>

#ifndef _GOURD_RAGGED_ARRAY_
#define _GOURD_RAGGED_ARRAY_

namespace gourd {

  /*! Ragged array 
   * Template alias for 
   * \c template<class T> \c std::vector<std::vector<T>>
   */
  template< typename T >
  using ragged_array = std::vector< std::vector<T> >;
  
}  // namespace gourd

#endif  // _GOURD_RAGGED_ARRAY_

