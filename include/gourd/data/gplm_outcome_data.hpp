
#include <cassert>
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
#include "gourd/nifti2.hpp"
#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/utilities/csv_reader.hpp"



#ifndef _GOURD_GPLM_OUTCOME_DATA_
#define _GOURD_GPLM_OUTCOME_DATA_


namespace gourd {

  /* ****************************************************************/
  /*! Outcome data for gourd gaussian process models
   */
  template< typename T >
  class gplm_outcome_data {
  public:
    typedef T scalar_type;
    typedef typename abseil::cartesian_coordinate<3, T>
      coord_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;

    gplm_outcome_data(
      const std::vector<std::string>& yfiles,
      const std::string surf_file
    );

    mat_type& y_ref() ;

    int n() const;
    int nloc() const;
    const mat_type&    y() const;
    const vector_type& y( const int i ) const;
    scalar_type        y( const int i, const int s ) const;
    const vector_type& yssq() const;

    const std::vector<coord_type>& coordinates() const;

    void center(); /*!< Mean center outcome vectors */

  protected:
    mat_type y_;
    vector_type yssq_;

    std::vector<coord_type> coord_;

    void read_data_(
      const std::vector<std::string>& yfiles,
      ::gifti_image* surface
    );

    void process_images_(
      ::nifti_image* metric,
      const gourd::cifti_gifti_pair& cgp,
      const int i
    );
  };
  // class gplm_data
  
};
// namespace gourd




template< typename T >
gourd::gplm_outcome_data<T>::gplm_outcome_data(
  const std::vector<std::string>& yfiles,
  const std::string surf_file
) {
  if ( yfiles.empty() ) {
    throw std::domain_error("gplm_outcome_data: must provide outcome "
			    "data files");
  }
  ::gifti_image* surface = gourd::gifti::image_read( surf_file );
  read_data_( yfiles, surface );
  // 
  ::gifti_free_image( surface );
};



template< typename T >
int gourd::gplm_outcome_data<T>::n() const {
  return y_.cols();
};


template< typename T >
int gourd::gplm_outcome_data<T>::nloc() const {
  return coord_.size();
};



template< typename T >
typename gourd::gplm_outcome_data<T>::mat_type&
gourd::gplm_outcome_data<T>::y_ref() {
  return y_;
};


template< typename T >
const typename gourd::gplm_outcome_data<T>::mat_type&
gourd::gplm_outcome_data<T>::y() const {
  return y_;
};


template< typename T >
const typename gourd::gplm_outcome_data<T>::vector_type&
gourd::gplm_outcome_data<T>::y(const int i) const {
  assert(i >= 0 && i < y_.cols() &&
	 "gplm_outcome_data::y(const int) :  index out of scope" );
  //
  return y_.col(i);
};


template< typename T >
T gourd::gplm_outcome_data<T>::y(const int i, const int s) const {
  assert(i >= 0 && i < y_.cols() &&
	 "gplm_outcome_data::y(const int, const int) :  "
	 "index 'i' out of scope" );
  assert(s >= 0 && s < y_.rows() &&
	 "gplm_outcome_data::y(const int, const int) :  "
	 "index 's' out of scope" );
  //
  return y_.coeffRef(s, i);
};



template< typename T >
const typename gourd::gplm_outcome_data<T>::vector_type&
gourd::gplm_outcome_data<T>::yssq()
  const {
  return yssq_;
};



template< typename T >
const std::vector<typename gourd::gplm_outcome_data<T>::coord_type>&
gourd::gplm_outcome_data<T>::coordinates()
  const {
  return coord_;
};


template< typename T >
void gourd::gplm_outcome_data<T>::center() {
  const vector_type mu = y_.colwise().mean();
  y_.rowwise() -= mu.adjoint();
};




template< typename T >
void gourd::gplm_outcome_data<T>::read_data_(
  const std::vector<std::string>& yfiles,
  ::gifti_image* surface
) {
  std::map<std::string, int> metric_intents;
  for ( int i = 0; i < (int)yfiles.size(); i++ ) {
    ::nifti_image* metric = gourd::nifti2::image_read( yfiles[i] );
    if ( ::nifti_looks_like_cifti(metric) == 0 ) {
      std::cerr << "\t*** Warning: file "
		<< yfiles[i]
		<< " does not match expected CIFTI format\n";
    }
    gourd::cifti_gifti_pair cgp(metric, surface);
    //
    if ( i == 0 ) {
      // Assign coord_ and allocate y_ memory
      gourd::extract_coordinates(surface, cgp, coord_ );
      y_ = mat_type( coord_.size(), (int)yfiles.size() );
      yssq_ = vector_type::Zero( coord_.size() );
    }
    std::string intent = ::nifti_intent_string( metric->intent_code );
    metric_intents.try_emplace(intent, 0);
    metric_intents[intent]++;
    process_images_( metric, cgp, i );
    ::nifti_image_free( metric );
  }
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
    std::cout << "\t" << ip.first << ":\t" << ip.second << std::endl;
  }
};



template< typename T >
void gourd::gplm_outcome_data<T>::process_images_(
  ::nifti_image* metric,
  const gourd::cifti_gifti_pair& cgp,
  const int i
) {
  const vector_type yi = gourd::map_cifti_to_mat<T>(
    metric,
    cgp.cifti_indices().cbegin(),
    cgp.cifti_indices().cend(),
    cgp.cifti_array_dim()
  );
  y_.col(i) = yi;
  yssq_ += yi.cwiseAbs2();
};


#endif  // _GOURD_GPLM_OUTCOME_DATA_
