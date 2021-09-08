
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "gourd/data/gplm_outcome_data.hpp"
#include "gourd/utilities/ragged_array.hpp"


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

    gplm_full_data(
      const std::vector<std::string>& yfiles,
      const std::string surf_file,
      const std::string covar_file,
      const gourd::ragged_array<int>& indices
    );
    
    int p() const;
    int xrank() const;
    const mat_type& yu() const;
    const mat_type& x()  const;
    const mat_type& xsvd_u() const;
    const mat_type& xsvd_v() const;
    const vector_type& xsvd_d() const;

    const Eigen::BDCSVD<mat_type>& xsvd() const;

    const gourd::ragged_array<int>& varcomp_indices() const;

  private:
    mat_type x_;
    mat_type yu_;
    Eigen::BDCSVD<mat_type> xudv_;
    gourd::ragged_array<int> vc_indices_;

    bool validate_indices_() const;
  };

}  // namespace gourd


template< typename T >
gourd::gplm_full_data<T>::gplm_full_data(
  const std::vector<std::string>& yfiles,
  const std::string surf_file,
  const std::string covar_file,
  const gourd::ragged_array<int>& indices
) :
  gourd::gplm_outcome_data<T>(yfiles, surf_file)
{
  x_ = gourd::utilities::read_csv<T>( covar_file );
  if ( x_.rows() != (int)yfiles.size() ) {
    throw std::domain_error( "Design matrix rows not equal to length "
			     "of outcome files" );
  }

  if ( indices.empty() ) {
    // Default: each coefficient gets its own variance component
    vc_indices_.resize( this->p() );
    for ( size_t i = 0; i < vc_indices_.size(); i++ )
      vc_indices_[i] = std::vector<int>(1u, (int)i);
  }
  else {
    // Copy indices and validate
    vc_indices_ = indices;
    if ( !validate_indices_() ) {
      throw std::domain_error("Invalid variance component indices");
    }
  }

  xudv_.compute( x_,
    Eigen::DecompositionOptions::ComputeThinU |
    Eigen::DecompositionOptions::ComputeThinV
  );

  //
  yu_ = this->y() * xudv_.matrixU();
};




template< typename T >
int gourd::gplm_full_data<T>::p() const {
  return x_.cols();
};

template< typename T >
int gourd::gplm_full_data<T>::xrank() const {
  return xudv_.rank();
};


template< typename T >
const typename gourd::gplm_full_data<T>::mat_type&
gourd::gplm_full_data<T>::yu() const {
  return yu_;
};


template< typename T >
const typename gourd::gplm_full_data<T>::mat_type&
gourd::gplm_full_data<T>::x() const {
  return x_;
};


template< typename T >
const gourd::ragged_array<int>&
gourd::gplm_full_data<T>::varcomp_indices() const {
  return vc_indices_;
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



template< typename T >
bool gourd::gplm_full_data<T>::validate_indices_() const {
  const int p = x_.cols();
  std::vector<int> counts( p, 0 );
  bool ok = true;
  for ( auto& row : vc_indices_ ) {
    for ( int j : row ) {
      if (j < 0 || j >= p)  return false;
      counts[j]++;
    }
  }
  for ( int c : counts ) if ( c != 1 ) { ok = false; break; }
  if ( !ok ) {
    std::cerr << "Variance component errors:\n";
    for ( size_t i = 0; i < counts.size(); i++ ) {
      if ( counts[i] != 1 ) {
	std::cerr << "Index (" << i << ") used " << counts[i]
		  << " times\n";
      }
    } 
  }
  return ok;
};



#endif  // _GOURD_GPML_FULL_DATA_


