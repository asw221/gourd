
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SVD>

#include "gifti_io.h"
#include "nifti2_io.h"

#include "abseil/coordinates.hpp"
#include "abseil/covariance_functors.hpp"

#include "gourd/eigen_map_cifti.hpp"
#include "gourd/gifti.hpp"
#include "gourd/neighborhood_smooth.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/options.hpp"
#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/utilities/csv_reader.hpp"


// #ifdef GOURD_PROFILE_COMPUTATIONS
#include "abseil/timer.hpp"
// #endif


#ifndef _GOURD_GPLM_SMOOTHED_SSTAT_
#define _GOURD_GPLM_SMOOTHED_SSTAT_


namespace gourd {

  /* ****************************************************************/
  /*! Sufficient statistics for gourd gaussian process regression 
   */
  template< typename T >
  class gplm_smoothed_sstat {
  public:
    typedef T scalar_type;
    typedef typename abseil::cartesian_coordinate<3, T>
      coord_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;
    typedef typename Eigen::SparseMatrix<T> spmat_type;

    /*! (constructor) */
    gplm_smoothed_sstat() = default;

    template< size_t D >
    gplm_smoothed_sstat(
      const std::string xfile,
      const std::vector<std::string>& yfiles,
      const std::string surf_file,
      const abseil::covariance_functor<T, D>* const cov,
      const T sm_radius,
      const gourd::dist_code distance
    );

    // gplm_smoothed_sstat( const gourd::gplm_full_data<T>& other );

    // gplm_smoothed_sstat<T>& operator=( const gourd::gplm_full_data<T>& other );
    // gplm_smoothed_sstat<T>& operator=( const gourd::gplm_smoothed_sstat<T>& other );

    mat_type& yu_ref() ;

    int n() const;
    int nloc() const;
    int p() const;
    int xrank() const;
    const mat_type& yu() const;
    const mat_type& x()  const;
    const mat_type& xsvd_u() const;
    const mat_type& xsvd_v() const;
    const vector_type& xsvd_d() const;
    const vector_type& yssq() const;

    const Eigen::BDCSVD<mat_type>& xsvd() const;
    const std::vector<coord_type>& coordinates() const;

  protected:
    mat_type yu_;
    mat_type x_;
    spmat_type sm_;
    vector_type yssq_;

    Eigen::BDCSVD<mat_type> xudv_;

    std::vector<coord_type> coord_;

    void read_xfile_(
      const std::string xfile,
      const std::vector<std::string>& yfiles
    );
    
    void stream_data_(
      const std::vector<std::string>& yfiles,
      ::gifti_image* surface
    );
    
    virtual void process_images_(
      ::nifti_image* metric,
      ::gifti_image* surface,
      const int i
    );
  };
  // class gplm_smoothed_sstat
  
};
// namespace gourd




template< typename T >
template< size_t D >
gourd::gplm_smoothed_sstat<T>::gplm_smoothed_sstat(
  const std::string xfile,
  const std::vector<std::string>& yfiles,
  const std::string surf_file,
  const abseil::covariance_functor<T, D>* const cov,
  const T sm_radius,
  const gourd::dist_code distance
) {
  if ( yfiles.empty() ) {
    throw std::domain_error("gplm_smoothed_sstat: must provide outcome "
			    "data files");
  }
  read_xfile_( xfile, yfiles );
  //
  ::gifti_image* surface = gourd::gifti::image_read( surf_file );
  ::nifti_image* metric  = gourd::nifti2::image_read( yfiles[0] ); 
  const gourd::cifti_gifti_pair cgp(metric, surface);
  gourd::extract_coordinates( surface, cgp, coord_ );
  ::nifti_image_free( metric );
  //
  sm_ = gourd::compute_nnsmooth_mat(coord_, cov, sm_radius, distance);
  stream_data_( yfiles, surface );
  //
  ::gifti_free_image( surface );
};





template< typename T >
int gourd::gplm_smoothed_sstat<T>::n() const {
  return x_.rows();
};


template< typename T >
int gourd::gplm_smoothed_sstat<T>::nloc() const {
  return yu_.rows();
};


template< typename T >
int gourd::gplm_smoothed_sstat<T>::p() const {
  return x_.cols();
};


template< typename T >
int gourd::gplm_smoothed_sstat<T>::xrank() const {
  return xudv_.rank();
};


template< typename T >
typename gourd::gplm_smoothed_sstat<T>::mat_type&
gourd::gplm_smoothed_sstat<T>::yu_ref() {
  return yu_;
};


template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::mat_type&
gourd::gplm_smoothed_sstat<T>::yu()
  const {
  return yu_;
};

template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::mat_type&
gourd::gplm_smoothed_sstat<T>::x()
  const {
  return x_;
};



template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::mat_type&
gourd::gplm_smoothed_sstat<T>::xsvd_u()
  const {
  return xudv_.matrixU();
};


template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::mat_type&
gourd::gplm_smoothed_sstat<T>::xsvd_v()
  const {
  return xudv_.matrixV();
};


template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::vector_type&
gourd::gplm_smoothed_sstat<T>::xsvd_d()
  const {
  return xudv_.singularValues();
};


template< typename T >
const typename gourd::gplm_smoothed_sstat<T>::vector_type&
gourd::gplm_smoothed_sstat<T>::yssq()
  const {
  return yssq_;
};


template< typename T >
const Eigen::BDCSVD<typename gourd::gplm_smoothed_sstat<T>::mat_type>&
gourd::gplm_smoothed_sstat<T>::xsvd() const {
  return xudv_;
};


template< typename T >
const std::vector<typename gourd::gplm_smoothed_sstat<T>::coord_type>&
gourd::gplm_smoothed_sstat<T>::coordinates()
  const {
  return coord_;
};



template< typename T >
void gourd::gplm_smoothed_sstat<T>::read_xfile_(
  const std::string xfile,
  const std::vector<std::string>& yfiles
) {
  x_ = gourd::utilities::read_csv<T>( xfile );
  if ( x_.rows() != (int)yfiles.size() ) {
    throw std::domain_error( "Design matrix rows not equal to length "
			     "of outcome files" );
  }
  xudv_.compute(x_,
    Eigen::DecompositionOptions::ComputeThinU |
    Eigen::DecompositionOptions::ComputeThinV
  );
};


template< typename T >
void gourd::gplm_smoothed_sstat<T>::stream_data_(
  const std::vector<std::string>& yfiles,
  ::gifti_image* surface
) {
  // #ifdef GOURD_PROFILE_COMPUTATIONS
  abseil::timer::start();
  // #endif
  std::map<std::string, int> metric_intents;
  for ( int i = 0; i < (int)yfiles.size(); i++ ) {
    ::nifti_image* metric = gourd::nifti2::image_read( yfiles[i] );
    if ( ! ::nifti_looks_like_cifti(metric) ) {
      std::cerr << "\t*** Warning: file "
		<< yfiles[i]
		<< " does not match expected CIFTI format\n";
    }
    std::string intent = ::nifti_intent_string( metric->intent_code );
    metric_intents.try_emplace(intent, 0);
    metric_intents[intent]++;
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
  // Print intent information
  std::cout << "GIFTI intent(s):  {";
  if ( surface->numDA > 0 && surface->darray ) {
    for ( int i = 0; i < surface->numDA; i++ )
      std::cout << ::nifti_intent_string( surface->darray[i]->intent )
		<< ", ";
  }
  std::cout << "}\n";
  std::cout << "CIFTI intent(s):\n";
  for ( auto& ip : metric_intents ) {
    std::cout << "\t" << ip.first << " - " << ip.second << std::endl;
  }
};



template< typename T >
void gourd::gplm_smoothed_sstat<T>::process_images_(
  ::nifti_image* metric,
  ::gifti_image* surface,
  const int i
) {
  const gourd::cifti_gifti_pair cgp(metric, surface);
  const vector_type y = sm_ * gourd::map_cifti_to_mat<T>(
    metric,
    cgp.cifti_paired_indices().cbegin(),
    cgp.cifti_paired_indices().cend(),
    cgp.cifti_array_dim()
  );
  if ( yu_.rows() == 0 || yu_.cols() == 0 ) {
    yu_ = mat_type::Zero( y.size(), xudv_.matrixU().cols() );
    yssq_ = vector_type::Zero( y.size() );
  }
  yu_   += y * xudv_.matrixU().row(i);
  yssq_ += y.cwiseAbs2();
};


#endif  // _GOURD_GPLM_SMOOTHED_SSTAT_
