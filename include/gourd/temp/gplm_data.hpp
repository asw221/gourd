
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "abseil/coordinates.hpp"
#include "gifti_io.h"
#include "nifti2_io.h"

#include "gourd/eigen_map_cifti.hpp"
#include "gourd/gifti.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/utilities/csv_reader.hpp"


// #ifdef GOURD_PROFILE_COMPUTATIONS
#include "abseil/timer.hpp"
// #endif


#ifndef _GOURD_GPLM_DATA_
#define _GOURD_GPLM_DATA_


namespace gourd {

  /* ****************************************************************/
  /*! Full data for gourd gaussian process regression 
   */
  template< typename T >
  class gplm_data {
  public:
    typedef T scalar_type;
    typedef typename abseil::cartesian_coordinate<3, T>
      coord_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;

    gplm_data(
      const std::string xfile,
      const std::vector<std::string>& yfiles,
      const std::string surf_file
    );

    int n() const;
    int nloc() const;
    int p() const;
    const mat_type& y() const;
    const mat_type& yu() const;
    const mat_type& x()  const;
    const mat_type& xsvd_u() const;
    const mat_type& xsvd_v() const;
    const vector_type& xsvd_d() const;
    const vector_type& yssq() const;

    const Eigen::BDCSVD<mat_type>& xsvd() const;
    const std::vector<coord_type>& coordinates() const;

  private:
    mat_type y_;
    mat_type yu_;
    mat_type x_;
    vector_type yssq_;

    Eigen::BDCSVD<mat_type> xudv_;

    std::vector<coord_type> coord_;

    void read_data_(
      const std::vector<std::string>& yfiles,
      ::gifti_image* surface
    );

    void process_images_(
      ::nifti_image* metric,
      ::gifti_image* surface,
      const int i
    );
  };
  // class gplm_data
  
};
// namespace gourd




template< typename T >
gourd::gplm_data<T>::gplm_data(
  const std::string xfile,
  const std::vector<std::string>& yfiles,
  const std::string surf_file
) {
  if ( yfiles.empty() ) {
    throw std::domain_error("gplm_data: must provide outcome "
			    "data files");
  }
  
  x_ = gourd::utilities::read_csv<T>( xfile );
  if ( x_.rows() != (int)yfiles.size() ) {
    throw std::domain_error( "Design matrix rows not equal to length "
			     "of outcome files" );
  }
  xudv_.compute(x_,
    Eigen::DecompositionOptions::ComputeThinU |
    Eigen::DecompositionOptions::ComputeThinV
  );

  ::gifti_image* surface = gourd::gifti::image_read( surf_file );
  read_data_( yfiles, surface );

  ::gifti_free_image( surface );
};



template< typename T >
int gourd::gplm_data<T>::n() const {
  return x_.rows();
};


template< typename T >
int gourd::gplm_data<T>::nloc() const {
  return coord_.size();
};


template< typename T >
int gourd::gplm_data<T>::p() const {
  return xudv_.rank();
};




template< typename T >
const typename gourd::gplm_data<T>::mat_type&
gourd::gplm_data<T>::y()
  const {
  return y_;
};

template< typename T >
const typename gourd::gplm_data<T>::mat_type&
gourd::gplm_data<T>::x()
  const {
  return x_;
};


template< typename T >
const typename gourd::gplm_data<T>::mat_type&
gourd::gplm_data<T>::xsvd_u()
  const {
  return xudv_.matrixU();
};


template< typename T >
const typename gourd::gplm_data<T>::mat_type&
gourd::gplm_data<T>::xsvd_v()
  const {
  return xudv_.matrixV();
};


template< typename T >
const typename gourd::gplm_data<T>::vector_type&
gourd::gplm_data<T>::xsvd_d()
  const {
  return xudv_.singularValues();
};


template< typename T >
const typename gourd::gplm_data<T>::vector_type&
gourd::gplm_data<T>::yssq()
  const {
  return yssq_;
};


template< typename T >
const Eigen::BDCSVD<typename gourd::gplm_data<T>::mat_type>&
gourd::gplm_data<T>::xsvd() const {
  return xudv_;
};


template< typename T >
const std::vector<typename gourd::gplm_data<T>::coord_type>&
gourd::gplm_data<T>::coordinates()
  const {
  return coord_;
};




template< typename T >
void gourd::gplm_data<T>::read_data_(
  const std::vector<std::string>& yfiles,
  ::gifti_image* surface
) {
  // #ifdef GOURD_PROFILE_COMPUTATIONS
  abseil::timer::start();
  // #endif
  for ( int i = 0; i < (int)yfiles.size(); i++ ) {
    ::nifti_image* metric = gourd::nifti2::image_read( yfiles[i] );
    if ( ::nifti_looks_like_cifti(metric) == 0 ) {
      std::cerr << "\t*** Warning: file "
		<< yfiles[i]
		<< " does not match expected CIFTI format\n";
    }
    process_images_( metric, surface, i );
    ::nifti_image_free( metric );
  }
  // #ifdef GOURD_PROFILE_COMPUTATIONS
  abseil::timer::stop();
  std::cout << "\t<< Image streaming took: ~"
	    << (abseil::timer::diff() / 1e3 / yfiles.size())
	    << " (ms) per image >>\n"
	    << std::endl;
  // #endif
};



template< typename T >
void gourd::gplm_data<T>::process_images_(
  ::nifti_image* metric,
  ::gifti_image* surface,
  const int i
) {
  const gourd::cifti_gifti_pair cgp(metric, surface);
  if ( coord_.empty() ) {
    gourd::extract_coordinates( surface, cgp, coord_ );
  }
  const vector_type yi = gourd::map_cifti_to_mat<T>(
    metric,
    cgp.cifti_indices().cbegin(),
    cgp.cifti_indices().cend(),
    cgp.cifti_array_dim()
  );
  if ( y_.rows() == 0 || y_.cols() == 0 ) {
    y_ = mat_type( nloc(), n() );
    yu_ = mat_type::Zero( yi.size(), p() );
    yssq_ = vector_type::Zero( yi.size() );
  }
  y_.col(i) = yi;
  yu_   += yi * xudv_.matrixU().row(i);
  yssq_ += yi.cwiseAbs2();
};


#endif  // _GOURD_GPLM_DATA_
