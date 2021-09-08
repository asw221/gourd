
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "nifti2_io.h"

#include "gourd/nifti2.hpp"
// #include "gourd/cifti_xml.hpp"


#ifndef _GOURD_EIGEN_MAP_CIFTI_
#define _GOURD_EIGEN_MAP_CIFTI_

namespace gourd {

  /* ****************************************************************/
  /*! Extract CIFTI data array into an \c Eigen::Matrix
   * 
   * Template parameter \c T should always be specified and controls
   * scalar type in the output matrix
   *
   * @return  \c Eigen::Matrix<T> containing a deep copy of the
   *   CIFTI data array.
   */
  template< typename T >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat( const ::nifti_image* nim );

  
  
  /*! 
   * Subsets extracted data along either the row or column dimension
   */
  template< typename T, typename IndT >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat(
    const ::nifti_image* nim, /*!< Input image */
    const IndT sset_start,    /*!< Iterator to subset indices */
    const IndT sset_end,      /*!< Iterator to subset indices */
    const int sset_dim = 0    /*!< Apply subset to dimension 0 or 1 */
  );


  

namespace def {

  /* ****************************************************************/
  template<
    typename ImT,  // Native CIFTI data type
    typename T     // CIFTI data recast to type T
    >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat(
    const ::nifti_image* nim
  );


  /* ****************************************************************/  
  template<
    typename ImT,  // Native CIFTI data type
    typename T,    // Recast extracted CIFTI data to type T
    typename IndT  // Subset index iterator type
    >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat(
    const ::nifti_image* nim,
    const IndT subset_start,
    const IndT subset_end,
    const int subset_dim = 0
  );


  
  /* ****************************************************************/
  template< typename ImT, typename T, typename IndT >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat_row_subset(
    const ::nifti_image* nim,
    const IndT subset_start,
    const IndT subset_end
  );


  
  /* ****************************************************************/
  template< typename ImT, typename T, typename IndT >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  map_cifti_to_mat_col_subset(
    const ::nifti_image* nim,
    const IndT subset_start,
    const IndT subset_end
  );
  
};
  // namespace def
  
  
};
// namespace gourd







/* ******************************************************************/
template< typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::map_cifti_to_mat(
  const ::nifti_image* nim
) {
  if ( gourd::nifti2::uses_datatype<short>(nim) ) {
    return gourd::def::map_cifti_to_mat<short, T>(nim);
  }
  else if ( gourd::nifti2::uses_datatype<int>(nim) ) {
    return gourd::def::map_cifti_to_mat<int, T>(nim);
  }
  else if ( gourd::nifti2::uses_datatype<float>(nim) ) {
    return gourd::def::map_cifti_to_mat<float, T>(nim);
  }
  else if ( gourd::nifti2::uses_datatype<double>(nim) ) {
    return gourd::def::map_cifti_to_mat<double, T>(nim);
  }
  else if ( gourd::nifti2::uses_datatype<long double>(nim) ) {
    return gourd::def::map_cifti_to_mat<long double, T>(nim);
  }

  return gourd::def::map_cifti_to_mat<float, T>(nim);
};
/* ******************************************************************/



/* ******************************************************************/
template< typename T, typename IndT >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::map_cifti_to_mat(
  const ::nifti_image* nim,
  const IndT sset_start,
  const IndT sset_end,
  const int sset_dim
) {
  if ( gourd::nifti2::uses_datatype<short>(nim) ) {
    return gourd::def::map_cifti_to_mat<short, T, IndT>
      (nim, sset_start, sset_end, sset_dim);
  }
  else if ( gourd::nifti2::uses_datatype<int>(nim) ) {
    return gourd::def::map_cifti_to_mat<int, T, IndT>
      (nim, sset_start, sset_end, sset_dim);
  }
  else if ( gourd::nifti2::uses_datatype<float>(nim) ) {
    return gourd::def::map_cifti_to_mat<float, T, IndT>
      (nim, sset_start, sset_end, sset_dim);
  }
  else if ( gourd::nifti2::uses_datatype<double>(nim) ) {
    return gourd::def::map_cifti_to_mat<double, T, IndT>
      (nim, sset_start, sset_end, sset_dim);
  }
  else if ( gourd::nifti2::uses_datatype<long double>(nim) ) {
    return gourd::def::map_cifti_to_mat<long double, T, IndT>
      (nim, sset_start, sset_end, sset_dim);
  }

  return gourd::def::map_cifti_to_mat<float, T, IndT>
    (nim, sset_start, sset_end, sset_dim);
};
/* ******************************************************************/








/* ******************************************************************/
template< typename ImT, typename T >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::def::map_cifti_to_mat(
  const ::nifti_image* nim
) {
  const std::vector<int> dims = gourd::nifti2::get_dims(nim);
  const int ndims = dims.size();
  if ( ndims > 2 ) {
    throw std::domain_error("Don't know how to deal with CIFTI data "
			    "arrays with more than 2 dimensions");
  }
  else if ( ndims <= 0 ) {
    throw std::domain_error("CIFTI data lacks dimension info");
  }
  //
  const int nvox = nim->nvox;
  const int nx = ( ndims >= 1 ) ? dims[0] : 1;
  const int ny = ( ndims >= 2 ) ? dims[1] : 1;
  const ImT* const data_ptr = gourd::nifti2::get_data_ptr<ImT>(nim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_data(nx, ny);
  int i, j;
  for ( int v = 0; v < nvox; v++ ) {
    // Column-major order
    i = v % nx;
    j = (v / nx) % ny;
    m_data.coeffRef(i, j) = static_cast<T>( *(data_ptr + v) );
  }
  return m_data;
};
/* ******************************************************************/





/* ******************************************************************/
template< typename ImT, typename T, typename IndT >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::def::map_cifti_to_mat(
  const ::nifti_image* nim,
  const IndT subset_start,
  const IndT subset_end,
  const int subset_dim
) {
  static_assert( std::is_integral_v<typename IndT::value_type>,
		 "map_cifti_to_mat: subset only defined for integer types" );
  const int ndims = gourd::nifti2::get_dims(nim).size();
  assert( subset_dim < ndims &&
	  "map_cifti_to_mat: subset_dim too large" );
  assert( subset_dim >= 0 &&
	  "map_cifti_to_mat: negative subset dim" );
  if ( ndims > 2 ) {
    throw std::domain_error("Don't know how to deal with CIFTI data "
			    "arrays with more than 2 dimensions");
  }
  else if ( ndims <= 0 ) {
    throw std::domain_error("CIFTI data lacks dimension info");
  }
  //
  return ( subset_dim == 0 ) ?
    gourd::def::map_cifti_to_mat_row_subset<ImT, T, IndT>
    ( nim, subset_start, subset_end ) :
    gourd::def::map_cifti_to_mat_col_subset<ImT, T, IndT>
    ( nim, subset_start, subset_end );
};
/* ******************************************************************/




/* ******************************************************************/
template< typename ImT, typename T, typename IndT >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::def::map_cifti_to_mat_row_subset(
  const ::nifti_image* nim,
  const IndT subset_start,
  const IndT subset_end
) {
  const std::vector<int> dims = gourd::nifti2::get_dims(nim);
  assert( (*subset_start) >= 0 );
  assert( *(subset_end - 1) < dims[0] );
  //
  const int ndims = dims.size();
  const int nx = ( ndims >= 1 ) ? dims[0] : 1;
  const int ny = ( ndims >= 2 ) ? dims[1] : 1;
  const int dx = std::distance(subset_start, subset_end);
  //
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_data(dx, ny);
  const ImT* const data_ptr = gourd::nifti2::get_data_ptr<ImT>(nim);
  IndT v_it = subset_start;
  int stride;
  for ( int j = 0; j < ny; j++ ) {
    for ( int i = 0; i < dx; i++, ++v_it ) {
      // Column-major
      stride = nx * j + (*v_it);
      m_data.coeffRef(i, j) = *(data_ptr + stride);
    }
  }
  return m_data;
};
/* ******************************************************************/



/* ******************************************************************/
template< typename ImT, typename T, typename IndT >
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
gourd::def::map_cifti_to_mat_col_subset(
  const ::nifti_image* nim,
  const IndT subset_start,
  const IndT subset_end
) {
  const std::vector<int> dims = gourd::nifti2::get_dims(nim);
  assert( *subset_start >= 0 );
  assert( *(subset_end - 1) < dims[1] );
  //  
  const int ndims = dims.size();
  const int nx = ( ndims >= 1 ) ? dims[0] : 1;
  const int dy = std::distance(subset_start, subset_end);
  //
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_data(nx, dy);
  const ImT* const data_ptr = gourd::nifti2::get_data_ptr<ImT>(nim);
  IndT v_it = subset_start;
  int stride;
  for ( int j = 0; j < dy; j++, ++v_it ) {
    for ( int i = 0; i < nx; i++ ) {
      // Column-major
      stride = nx * (*v_it) + i;
      m_data.coeffRef(i, j) = *(data_ptr + stride);
    }
  }
  return m_data;
};
/* ******************************************************************/



#endif  // _GOURD_EIGEN_MAP_CIFTI_
