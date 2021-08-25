
#include <cassert>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#include "nifti2_io.h"

#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/output_log.hpp"


#ifndef _GOURD_OUTPUT_
#define _GOURD_OUTPUT_

namespace gourd {

  template< typename PtrT >
  bool emplace_paired_data(
    ::nifti_image* nim,
    const gourd::cifti_gifti_pair& cgp,
    const PtrT data_begin,
    const PtrT data_end
  );

  template< typename PtrT, typename ImgT >
  void emplace_paired_data_impl(
    ::nifti_image* nim,
    const gourd::cifti_gifti_pair& cgp,
    const PtrT data_begin,
    const PtrT data_end
  );


  /*! Create CIFTI image from \c Eigen::Matrix input 
   *
   * If the matrix has multiple columns, these are considered
   * to correspond with a series of CIFTI images. Conversely,
   * rows of the matrix are taken to correspond to CIFTI
   * vertices/pixels
   */  
  template< typename T, int _rows, int _cols,
  	    int _opts, int _maxr, int _maxc >
  ::nifti_image* matrix_to_cifti(
    const Eigen::Matrix<T, _rows,_cols,_opts,_maxr,_maxc>& mat,
    ::nifti_image* ref
  );

  
  template< typename T, int _rows, int _cols,
	    int _opts, int _maxr, int _maxc >
  bool write_matrix_to_cifti(
    const Eigen::Matrix<T, _rows,_cols,_opts,_maxr,_maxc>& mat,
    ::nifti_image* ref,
    const std::string& fname
  );
  
  
}
// namespace gourd





 template< typename T, int _rows, int _cols,
	   int _opts, int _maxr, int _maxc >
::nifti_image* gourd::matrix_to_cifti(
  const Eigen::Matrix<T, _rows,_cols,_opts,_maxr,_maxc>& mat,
  ::nifti_image* ref
) {
  assert( ref->nv == mat.rows() &&
    "matrix_to_cifti: matrix rows != reference CIFTI dimension" );
  assert( ::nifti_looks_like_cifti(ref) &&
    "matrix_to_cifti: reference image not in CIFTI format" );
  namespace nii = gourd::nifti2;
  const int nc = mat.cols();
  ::nifti_image* outnim = nii::create_cifti(
    ref, nc, nii::intent::estimate, nii::data_t<T> );
  T* const data_ptr = static_cast<T*>( outnim->data );
  for ( int i = 0; i < mat.rows(); i++ ) {
    int stride = i * nc;
    for ( int j = 0; j < nc; j++, stride++ )
      *(data_ptr + stride) = mat.coeffRef(i, j);
  }
  return outnim;
};


template< typename T, int _rows, int _cols,
	  int _opts, int _maxr, int _maxc >
bool gourd::write_matrix_to_cifti(
  const Eigen::Matrix<T, _rows,_cols,_opts,_maxr,_maxc>& mat,
  ::nifti_image* ref,
  const std::string& fname
) {
  bool success = true;
  try {
    ::nifti_image* nim = gourd::matrix_to_cifti( mat, ref );
    gourd::nifti2::image_write( nim, fname );
    ::nifti_image_free( nim );
  }
  catch ( ... ) {
    success = false;
  }
  return success;
};




template< typename PtrT, typename ImgT >
void gourd::emplace_paired_data_impl(
  ::nifti_image* nim,
  const gourd::cifti_gifti_pair& cgp,
  const PtrT data_begin,
  const PtrT data_end
) {
  assert( std::distance(data_begin, data_end) ==
	  cgp.cifti_paired_indices().size() &&
	  "emplace_paired_data: bad CIFTI index mapping" );
  ImgT* data = (ImgT*)nim->data;
  const int nvox = nim->nvox;
  for ( int i = 0; i < nvox; i++ )
    *(data + i) = (ImgT)0;
  int i = 0;
  for ( PtrT ptr = data_begin; ptr != data_end; ++ptr ) {
    int stride = cgp.cifti_paired_indices()[i];
    *(data + stride) = static_cast<ImgT>( *ptr );
    i++;
  }
};


template< typename PtrT >
bool gourd::emplace_paired_data(
  ::nifti_image* nim,
  const gourd::cifti_gifti_pair& cgp,
  const PtrT data_begin,
  const PtrT data_end
) {
  bool success = true;
  try {
    if ( gourd::nifti2::uses_datatype<float>(nim) ) {
      gourd::emplace_paired_data_impl<PtrT, float>
	(nim, cgp, data_begin, data_end);
    }
    else if ( gourd::nifti2::uses_datatype<double>(nim) ) {
      gourd::emplace_paired_data_impl<PtrT, double>
	(nim, cgp, data_begin, data_end);
    }
    else {
      // datatype not implemented
      success = false;
    }
  }
  catch (...) {
    success = false;
  }
  return success;
};

#endif  // _GOURD_OUTPUT_

