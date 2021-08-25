
#include <iostream>
#include <map>
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
// #include "gourd/gplm_full_data.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/utilities/csv_reader.hpp"


// #ifdef GOURD_PROFILE_COMPUTATIONS
#include "abseil/timer.hpp"
// #endif


#ifndef _GOURD_GPLM_SSTAT_
#define _GOURD_GPLM_SSTAT_


namespace gourd {

  /* ****************************************************************/
  /*! Sufficient statistics for gourd gaussian process regression 
   */
  template< typename T >
  class gplm_sstat {
  public:
    typedef T scalar_type;
    typedef typename abseil::cartesian_coordinate<3, T>
      coord_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;

    gplm_sstat(
      const std::string xfile,
      const std::vector<std::string>& yfiles,
      const std::string surf_file
    );

    // gplm_sstat( const gourd::gplm_full_data<T>& other );

    // gplm_sstat<T>& operator=( const gourd::gplm_full_data<T>& other );
    // gplm_sstat<T>& operator=( const gourd::gplm_sstat<T>& other );

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

  private:
    mat_type yu_;
    mat_type x_;
    vector_type yssq_;

    Eigen::BDCSVD<mat_type> xudv_;

    std::vector<coord_type> coord_;

    void stream_data_(
      const std::vector<std::string>& yfiles,
      ::gifti_image* surface
    );

    void process_images_(
      ::nifti_image* metric,
      ::gifti_image* surface,
      const int i
    );
  };
  // class gplm_sstat
  
};
// namespace gourd




template< typename T >
gourd::gplm_sstat<T>::gplm_sstat(
  const std::string xfile,
  const std::vector<std::string>& yfiles,
  const std::string surf_file
) {
  if ( yfiles.empty() ) {
    throw std::domain_error("gplm_sstat: must provide outcome "
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
  stream_data_( yfiles, surface );

  ::gifti_free_image( surface );
};



// template< typename T >
// gourd::gplm_sstat<T>::gplm_sstat(
//   const gourd::gplm_full_data<T>& other
// ) {
//   x_ = other.x();
//   yu_ = other.y() * other.xsvd_u();
//   yssq_ = other.yssq();
//   xudv_.compute(x_,
//     Eigen::DecompositionOptions::ComputeThinU |
//     Eigen::DecompositionOptions::ComputeThinV
//   );
//   coord_ = other.coordinates();
// };


// template< typename T >
// gourd::gplm_sstat<T>& gourd::gplm_sstat<T>::operator=(
//   const gourd::gplm_full_data<T>& other
// ) {
//   x_ = other.x();
//   yu_ = other.y() * other.xsvd_u();
//   yssq_ = other.yssq();
//   xudv_.compute(x_,
//     Eigen::DecompositionOptions::ComputeThinU |
//     Eigen::DecompositionOptions::ComputeThinV
//   );
//   coord_ = other.coordinates();
// };


// template< typename T >
// gourd::gplm_sstat<T>& gourd::gplm_sstat<T>::operator=(
//   const gourd::gplm_sstat<T>& other
// ) {
//   x_ = other.x();
//   yu_ = other.yu();
//   yssq_ = other.yssq();
//   xudv_.compute(x_,
//     Eigen::DecompositionOptions::ComputeThinU |
//     Eigen::DecompositionOptions::ComputeThinV
//   );
// };



template< typename T >
int gourd::gplm_sstat<T>::n() const {
  return x_.rows();
};


template< typename T >
int gourd::gplm_sstat<T>::nloc() const {
  return yu_.rows();
};


template< typename T >
int gourd::gplm_sstat<T>::p() const {
  return x_.cols();
};


template< typename T >
int gourd::gplm_sstat<T>::xrank() const {
  return xudv_.rank();
};


template< typename T >
typename gourd::gplm_sstat<T>::mat_type&
gourd::gplm_sstat<T>::yu_ref() {
  return yu_;
};


template< typename T >
const typename gourd::gplm_sstat<T>::mat_type&
gourd::gplm_sstat<T>::yu()
  const {
  return yu_;
};

template< typename T >
const typename gourd::gplm_sstat<T>::mat_type&
gourd::gplm_sstat<T>::x()
  const {
  return x_;
};



template< typename T >
const typename gourd::gplm_sstat<T>::mat_type&
gourd::gplm_sstat<T>::xsvd_u()
  const {
  return xudv_.matrixU();
};


template< typename T >
const typename gourd::gplm_sstat<T>::mat_type&
gourd::gplm_sstat<T>::xsvd_v()
  const {
  return xudv_.matrixV();
};


template< typename T >
const typename gourd::gplm_sstat<T>::vector_type&
gourd::gplm_sstat<T>::xsvd_d()
  const {
  return xudv_.singularValues();
};


template< typename T >
const typename gourd::gplm_sstat<T>::vector_type&
gourd::gplm_sstat<T>::yssq()
  const {
  return yssq_;
};


template< typename T >
const Eigen::BDCSVD<typename gourd::gplm_sstat<T>::mat_type>&
gourd::gplm_sstat<T>::xsvd() const {
  return xudv_;
};


template< typename T >
const std::vector<typename gourd::gplm_sstat<T>::coord_type>&
gourd::gplm_sstat<T>::coordinates()
  const {
  return coord_;
};




template< typename T >
void gourd::gplm_sstat<T>::stream_data_(
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
void gourd::gplm_sstat<T>::process_images_(
  ::nifti_image* metric,
  ::gifti_image* surface,
  const int i
) {
  const gourd::cifti_gifti_pair cgp(metric, surface);
  if ( coord_.empty() ) {
    gourd::extract_coordinates( surface, cgp, coord_ );
  }
  const vector_type y = gourd::map_cifti_to_mat<T>(
    metric,
    cgp.cifti_indices().cbegin(),
    cgp.cifti_indices().cend(),
    cgp.cifti_array_dim()
  );
  if ( yu_.rows() == 0 || yu_.cols() == 0 ) {
    yu_ = mat_type::Zero( y.size(), xudv_.matrixU().cols() );
    yssq_ = vector_type::Zero( y.size() );
  }
  yu_   += y * xudv_.matrixU().row(i);
  yssq_ += y.cwiseAbs2();
};


#endif  // _GOURD_GPLM_SSTAT_
