
// #include <complex>
#include <nifti1.h>


#ifndef _GOURD_NIFTI_DATA_TYPE_
#define _GOURD_NIFTI_DATA_TYPE_


namespace gourd {


  
  template< int T >
  class nifti_data_type {
    typedef void  type;
    typedef void* pointer;
  };

  

  template<>
  class nifti_data_type< NIFTI_TYPE_INT16 > {
    typedef short   type;
    typedef short * pointer;
  };
  
  template<>
  class nifti_data_type< NIFTI_TYPE_INT32 > {
    typedef int   type;
    typedef int * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_FLOAT32 > {
    typedef float   type;
    typedef float * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_FLOAT64 > {
    typedef double   type;
    typedef double * pointer;
  };

  

  template<>
  class nifti_data_type< NIFTI_TYPE_UINT16 > {
    typedef unsigned short   type;
    typedef unsigned short * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_UINT32 > {
    typedef unsigned int   type;
    typedef unsigned int * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_INT64 > {
    typedef long long int   type;
    typedef long long int * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_UINT64 > {
    typedef unsigned long long int   type;
    typedef unsigned long long int * pointer;
  };

  template<>
  class nifti_data_type< NIFTI_TYPE_FLOAT128 > {
    typedef long double   type;
    typedef long double * pointer;
  };


  
    
};

#endif  // _GOURD_NIFTI_DATA_TYPE_




/*

  template<>
  struct nifti_data_type< NIFTI_TYPE_UINT8 > {
    typedef unsigned char   type;
    typedef unsigned char * pointer;
  };

  template<>
  struct nifti_data_type< NIFTI_TYPE_COMPLEX64 > {
    typedef std::complex<float>   type;
    typedef std::complex<float> * pointer;
  };

  template<>
  struct nifti_data_type< NIFTI_TYPE_RGB24 >;

  template<>
  struct nifti_data_type< NIFTI_TYPE_INT8 >;

  template<>
  struct nifti_data_type< NIFTI_TYPE_COMPLEX128 > {
    typedef std::complex<double>   type;
    typedef std::complex<double> * pointer;
  };

  template<>
  struct nifti_data_type< NIFTI_TYPE_COMPLEX256 > {
    typedef std::complex<long double>   type;
    typedef std::complex<long double> * pointer;
  };

  template<>
  struct nifti_data_type< NIFTI_TYPE_RGBA32 >;

*/
