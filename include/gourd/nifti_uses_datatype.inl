

#ifndef _GOURD_NIFTI_USES_DATATYPE_
#define _GOURD_NIFTI_USES_DATATYPE_


#include "nifti1.h"
#include "nifti2_io.h"

#include "gourd/nifti2.hpp"



/* Default: return false for any datatype code without a specific
 * implementation.
 *
 * Type codes defined in header "nifti1.h"; some types have
 * alias codes
 */
template< typename T >
bool gourd::nifti2::uses_datatype(
  const ::nifti_image* const nim
) {
  return false;
};



template<>
bool gourd::nifti2::uses_datatype<short>(
  const ::nifti_image* const nim
) {
  const int code = nim->datatype;
  return code == DT_SIGNED_SHORT ||
    code == DT_INT16;
};


template<>
bool gourd::nifti2::uses_datatype<int>(
  const ::nifti_image* const nim
) {
  const int code = nim->datatype;
  return code == DT_SIGNED_INT ||
    code == DT_INT32;
};


template<>
bool gourd::nifti2::uses_datatype<float>(
  const ::nifti_image* const nim
) {
  const int code = nim->datatype;
  return code == DT_FLOAT ||
    code == DT_FLOAT32;
};


template<>
bool gourd::nifti2::uses_datatype<double>(
  const ::nifti_image* const nim
) {
  const int code = nim->datatype;
  return code == DT_DOUBLE ||
    code == DT_FLOAT64;
};


template<>
bool gourd::nifti2::uses_datatype<long double>(
  const ::nifti_image* const nim
) {
  const int code = nim->datatype;
  return code == DT_FLOAT128;
};



#endif  // _GOURD_NIFTI_USES_DATATYPE_
