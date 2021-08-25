
#include <cassert>
#include <iterator>
#include <memory>

#include "abseil/covariance_functors.hpp"

#include "gourd/options.hpp"


#ifndef _GOURD_COVARIANCE_
#define _GOURD_COVARIANCE_

namespace gourd {
  
template< typename T, size_t M, typename InputIt >
void init_cov(
  std::unique_ptr< abseil::covariance_functor<T, M> >& cov_ptr,
  const gourd::cov_code function_code,
  InputIt first,
  InputIt last
);
  
}  // namespace gourd




template< typename T, size_t M, typename InputIt >
void gourd::init_cov(
  std::unique_ptr< abseil::covariance_functor<T, M> >& cov_ptr,
  const gourd::cov_code function_code,
  InputIt first,
  InputIt last
) {
  assert( std::distance(first, last) == M &&
	  "init_cov: bad number of parameters" );
  using rbf_t = abseil::radial_basis<T>;
  using matern_t = abseil::matern<T>;
  switch (function_code) {
    case gourd::cov_code::rbf : {
      cov_ptr = std::make_unique<rbf_t>( first, last );
      break;
    }
    case gourd::cov_code::matern : {
      cov_ptr = std::make_unique<matern_t>( first, last );
      break;
    }
  };
};

#endif  // _GOURD_COVARIANCE_
