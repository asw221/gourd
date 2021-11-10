
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


  template< typename T, size_t M >
  gourd::cov_code get_cov_code(
    const abseil::covariance_functor<T, M>* const cov_ptr
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
  using rq_t = abseil::rational_quadratic<T>;
  using matern_t = abseil::matern<T>;
  switch (function_code) {
    case gourd::cov_code::rbf : {
      cov_ptr = std::make_unique<rbf_t>( first, last ); break;
    }
    case gourd::cov_code::rq : {
      cov_ptr = std::make_unique<rq_t>( first, last ); break;
    }
    case gourd::cov_code::matern : {
      cov_ptr = std::make_unique<matern_t>( first, last ); break;
    }
  };
};



template< typename T, size_t M >
gourd::cov_code gourd::get_cov_code(
  const abseil::covariance_functor<T, M>* const cov_ptr
) {
  // using rbf_t = abseil::radial_basis<T>;
  using rq_t = abseil::rational_quadratic<T>;
  using matern_t = abseil::matern<T>;
  gourd::cov_code code = gourd::cov_code::rbf;
  if ( dynamic_cast<const rq_t* const>(cov_ptr) != NULL ) {
    code = gourd::cov_code::rq;
  }
  else if ( dynamic_cast<const matern_t* const>(cov_ptr) != NULL ) {
    code = gourd::cov_code::matern;
  }
  return code;
};


#endif  // _GOURD_COVARIANCE_
