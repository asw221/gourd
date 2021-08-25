
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "gourd/gplm_outcome_data.hpp"


#ifndef _GOURD_GPML_FULL_DATA_
#define _GOURD_GPML_FULL_DATA_

namespace gourd {

  template< typename T >
  class gplm_full_data :
    public gourd::gplm_outcome_data<T> {
  public:
    using scalar_type = typename
      gourd::gplm_outcome_data<T>::scalar_type;
    using coord_type  = typename
      gourd::gplm_outcome_data<T>::coord_type;
    using mat_type    = typename
      gourd::gplm_outcome_data<T>::mat_type;
    using vector_type = typename
      gourd::gplm_outcome_data<T>::vector_type;
    // using gourd::gplm_outcome_data<T>::y_ref;
    // using gourd::gplm_outcome_data<T>::n;
    // using gourd::gplm_outcome_data<T>::nloc;
    // using gourd::gplm_outcome_data<T>::y;
    // using gourd::gplm_outcome_data<T>::yssq;
    // using gourd::gplm_outcome_data<T>::coordinates;

    gplm_full_data(
      const std::vector<std::string>& yfiles,
      const std::string surf_file,
      const std::string covar_file
    );
    
    int p() const;
    const mat_type& x()  const;
    const mat_type& xsvd_u() const;
    const mat_type& xsvd_v() const;
    const vector_type& xsvd_d() const;

    const Eigen::BDCSVD<mat_type>& xsvd() const;

  private:
    mat_type x_;
    Eigen::BDCSVD<mat_type> xudv_;
  };

}  // namespace gourd


template< typename T >
gourd::gplm_full_data<T>::gplm_full_data(
      const std::vector<std::string>& yfiles,
      const std::string surf_file,
      const std::string covar_file
) :
  gourd::gplm_outcome_data<T>(yfiles, surf_file)
{
  x_ = gourd::utilities::read_csv<T>( covar_file );
  if ( x_.rows() != yfiles.size() ) {
    throw std::domain_error( "Design matrix rows not equal to length "
			     "of outcome files" );
  }

  xudv_.compute( x_,
    Eigen::DecompositionOptions::ComputeThinU |
    Eigen::DecompositionOptions::ComputeThinV
  );
};




template< typename T >
int gourd::gplm_full_data<T>::p() const {
  return xudv_.rank();
};

template< typename T >
const typename gourd::gplm_full_data<T>::mat_type&
gourd::gplm_full_data<T>::x()  const {
  return x_;
};

template< typename T >
const typename gourd::gplm_full_data<T>::mat_type&
gourd::gplm_full_data<T>::xsvd_u() const {
  return xudv_.matrixU();
};

template< typename T >
const typename gourd::gplm_full_data<T>::mat_type&
gourd::gplm_full_data<T>::xsvd_v() const {
  return xudv_.matrixV();
};

template< typename T >
const typename gourd::gplm_full_data<T>::vector_type&
gourd::gplm_full_data<T>::xsvd_d() const {
  return xudv_.singularValues();
};

template< typename T >
const Eigen::BDCSVD<typename gourd::gplm_full_data<T>::mat_type>&
gourd::gplm_full_data<T>::xsvd() const {
  return xudv_;
};

#endif  // _GOURD_GPML_FULL_DATA_


