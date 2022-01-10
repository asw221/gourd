
#include <array>
#include <cassert>
#include <cstdio>  // remove
#include <filesystem>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
// #include <variant>

#include "nifti/nifti1.h"
#include "nifti/nifti2_io.h"

// #include <memory>

#ifndef _GOURD_NIFTI_2_
#define _GOURD_NIFTI_2_

namespace gourd {
namespace nifti2 {
    

  /*! NIfTI intent codes
   *
   * Descriptions of the meaning of values stored in each voxel.
   * Distribution parameters are provided in the p1, ... p[n] fields.
   * See \c nifti1.h for more information
   */
  enum class intent {
    none = NIFTI_INTENT_NONE,
    /* Statistics */
    correl      = NIFTI_INTENT_CORREL,     /*!< Correlations */
    ttest       = NIFTI_INTENT_TTEST,      /*!< Student-t */
    ftest       = NIFTI_INTENT_FTEST,      /*!< F distribution */
    zscore      = NIFTI_INTENT_ZSCORE,     /* Std. normal */
    chisq       = NIFTI_INTENT_CHISQ,      /*!< Chi^2 */
    beta        = NIFTI_INTENT_BETA,       /*!< Beta distribution */
    binom       = NIFTI_INTENT_BINOM,      /*!< Binomial */
    gamma       = NIFTI_INTENT_GAMMA,      /*!< Gamma */
    poisson     = NIFTI_INTENT_POISSON,    /*!< Poisson */
    normal      = NIFTI_INTENT_NORMAL,     /*!< Gaussian */
    ftest_nonc  = NIFTI_INTENT_FTEST_NONC, /*!< Noncentral F */
    chisq_nonc  = NIFTI_INTENT_CHISQ_NONC, /*!< Noncentral chi^2 */
    logistic    = NIFTI_INTENT_LOGISTIC,   /*!< Logistic */
    laplace     = NIFTI_INTENT_LAPLACE,    /*!< Laplace */
    uniform     = NIFTI_INTENT_UNIFORM,    /*!< Uniform */
    ttest_nonc  = NIFTI_INTENT_TTEST_NONC, /*!< Noncentral Student-t */
    weibull     = NIFTI_INTENT_WEIBULL,    /*!< Weibull */
    chi         = NIFTI_INTENT_CHI,        /*!< Chi values*/
    invgauss    = NIFTI_INTENT_INVGAUSS,   /*!< Inverse Gaussian */
    extval      = NIFTI_INTENT_EXTVAL,     /*!< Extreme value distribution */
    pval        = NIFTI_INTENT_PVAL,       /*!< P-values */
    logpval     = NIFTI_INTENT_LOGPVAL,    /*!< log(p-values) */
    log10pval   = NIFTI_INTENT_LOG10PVAL,  /*!< log10(p-values) */
    first_statcode = NIFTI_FIRST_STATCODE, /*!< i.e. correlation */
    last_statcode  = NIFTI_LAST_STATCODE,  /*!< i.e. log10pval */
    /* Non-statistics */
    estimate    = NIFTI_INTENT_ESTIMATE,   /*!< Parameter estimates */
    label       = NIFTI_INTENT_LABEL,      /*<! Indices into labels */
    neuroname   = NIFTI_INTENT_NEURONAME,  /*!< NeuroNames labels indices */
    genmatrix   = NIFTI_INTENT_GENMATRIX,  /*!< MxN matrices */
    symmatrix   = NIFTI_INTENT_SYMMATRIX,  /*!< NxN symmetric matrices */
    dispvect    = NIFTI_INTENT_DISPVECT,   /*!< Displacements */
    vector      = NIFTI_INTENT_VECTOR,     /*!< Other vectors */
    pointset    = NIFTI_INTENT_POINTSET,   /*!< Spatial coordinates */
    triangle    = NIFTI_INTENT_TRIANGLE,   /*!< Index triplets */
    quaternion  = NIFTI_INTENT_QUATERNION, /*!< Quaternion */
    dimless     = NIFTI_INTENT_DIMLESS,    /*!< Generic other */
    /* GIFTI-specific */
    time_series = NIFTI_INTENT_TIME_SERIES,
    node_index  = NIFTI_INTENT_NODE_INDEX,
    rgb_vector  = NIFTI_INTENT_RGB_VECTOR,
    rgba_vector = NIFTI_INTENT_RGBA_VECTOR,
    shape       = NIFTI_INTENT_SHAPE,
    /* FSL Extensions */
#ifdef NIFTI_INTENT_FSL_FNIRT_DISPLACEMENT_FILED
    fsl_dfield  = NIFTI_INTENT_FSL_FNIRT_DISPLACEMENT_FIELD,
    fsl_cspline = NIFTI_INTENT_FSL_CUBIC_SPLINE_COEFFICIENTS,
    fsl_dct     = NIFTI_INTENT_FSL_DCT_COEFFICIENTS,
    fsl_qspline = NIFTI_INTENT_FSL_QUADRATIC_SPLINE_COEFFICIENTS,
    fsl_topup_cspline = NIFTI_INTENT_FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS,
    fsl_topup_qspline = NIFTI_INTENT_FSL_TOPUP_QUADRATIC_SPLINE_COEFFICIENTS,
    fsl_topup_field   = NIFTI_INTENT_FSL_TOPUP_FIELD
#endif
  };


  template< typename T > inline constexpr  int data_t  = 0;
  template<> inline constexpr  int data_t<float>       = 16;  
  template<> inline constexpr  int data_t<double>      = 64;  
  template<> inline constexpr  int data_t<long double> = 1536;  
  template<> inline constexpr  int data_t<short>       = 4;  
  template<> inline constexpr  int data_t<int>         = 8;


  /*! Read NIfTI file
   *
   * @param fname  Path to NIfTI file
   * @param read   Binary code: \c 0 will read just the header
   *               information. Otherwise the NIfTI data array will
   *               be read as well
   * @return  Pointer to a \c nifti_image data structure. Will need
   *   to be freed with \c ::nifti_image_free()
   */
  ::nifti_image* image_read(
    const std::string fname,
    const int read = 1
  );

  
  /*! Write NIfTI file
   * 
   * Overwrites data if fname already exists
   */
  void image_write(
    ::nifti_image* nim,
    std::string fname = ""
  );


  /*! Create new CIFTI image by example
   */
  ::nifti_image* create_cifti(
    ::nifti_image* src,
    const int nu,
    const intent ic = intent::estimate,
    const int dtype = data_t<float>
  );
  
  /*! Create new CIFTI image by dimensions
   */
  ::nifti_image* create_cifti(
    const int nu,
    const int nv,
    const intent ic = intent::estimate,
    const int dtype = data_t<float>
  );


  /*! Get dimensions of data array 
   * 
   * First position is 'ndim' field (last dimension greater than 1)
   */
  std::array<int64_t, 8> dim( const ::nifti_image* const nim );
  

  /*! Type-cast pointer to raw image data
   *
   * It is entirely up to the user to make sure they have the correct
   * underlying image type \c T.
   */
  template< typename T >
  inline T* get_data_ptr( const ::nifti_image* const nim );

  

  /*! Get dimensions of data array
   *
   * DISCARDS singleton dimensions
   */
  std::vector<int> get_dims( const ::nifti_image* const nim );
  
  
  /*! Check \c nifti_image \c datatype field 
   *
   * Returns \c true if the \c datatype field matches the template
   * parameter type \c T, and \c false otherwise
   *
   * Only implemented for the real numeric scalar types,
   *   - \c short
   *   - \c int
   *   - \c float
   *   - \c double
   *   - \c long double 
   */
  template< typename T >
  bool uses_datatype( const ::nifti_image* const nim );


  /*! Replace CIFTI extension
   *
   * Mildly dangerous operation
   */
  void replace_cifti_extension(
    ::nifti_image* const nim,
    const std::string& ext
  );
  
};
  // namespace gourd::nifti2
  
};
// namespace gourd






::nifti_image* gourd::nifti2::image_read(
  const std::string fname,
  const int read
) {
  const std::filesystem::path initial_dir =
    std::filesystem::current_path();
  const std::filesystem::path fp( fname );
  ::nifti_image* nim = NULL;
  try {
    std::filesystem::current_path( fp.parent_path() );
    nim = ::nifti_image_read( fp.filename().c_str(), read );
  }
  catch ( const std::exception& ex ) {
    std::cerr << "\t*** Error reading "
	      << fname
	      << ":\n\t"
	      << ex.what()
	      << std::endl;
  }
  std::filesystem::current_path( initial_dir );
  return nim;
};



void gourd::nifti2::image_write(
  ::nifti_image* nim,
  std::string fname
) {
  const std::filesystem::path initial_dir =
    std::filesystem::current_path();
  if ( fname.empty() ) {
    fname = std::string(nim->fname);
  }
  const std::filesystem::path fp( fname );
  try {
    if ( !fp.parent_path().empty() ) {
      std::filesystem::current_path( fp.parent_path() );
    }
    remove( fp.filename().c_str() );
    ::nifti_set_filenames(nim, fp.filename().c_str(), 1, 1);
    ::nifti_image_write(nim);
  }
  catch ( const std::exception& ex ) {
    std::cerr << "\t*** Error writing "
	      << fname
	      << ":\n\t"
	      << ex.what()
	      << std::endl;
  }
  std::filesystem::current_path( initial_dir );
};







::nifti_image* gourd::nifti2::create_cifti(
  ::nifti_image* src,
  const int nu,
  const gourd::nifti2::intent ic,
  const int dtype
) {
  assert( nu >= 1 && "create_cifti: CIFTI dims must be > 0" );
  if ( !::nifti_looks_like_cifti(src) ) {
    throw std::domain_error("create_cifti: source image does not "
			    "match expected CIFTI form");
  }
  std::array<int64_t, 8> dims = gourd::nifti2::dim(src);
  dims[ 5 ] = nu;
  ::nifti_image* nim = ::nifti_make_new_nim(dims.data(), dtype, 1);
  nim->intent_code = static_cast<int>(ic);
  if ( src->num_ext > 0 && src->ext_list ) {
    ::nifti_copy_extensions( nim, src );
    /* Edit extension data
     *  - characters are often multibyte, so beware
     */
    std::regex ex(
      "(<MatrixIndicesMap.*AppliesToMatrixDimension=\"?0\"?)"
      "(.*NumberOfSeriesPoints=)\"?[0-9]+\"?"
      "(.*)/>" );
    std::string oldext = nim->ext_list->edata;
    std::ostringstream formatss;
    std::smatch sm;  // Nearly unused
    if ( !std::regex_search( oldext, sm, ex ) ) {
      throw std::runtime_error(
        "create_cifti: failed to find expected token in XML" );
    }
    formatss << "$1"                  // AppliesToMatrixDimension
	     << "$2\"" << nu << "\""  // NumberOfSeriesPoints
	     << "$3"                  // Etc.
	     << "/>";                 // XML block terminus
    std::string ext = std::regex_replace(oldext, ex, formatss.str());
    gourd::nifti2::replace_cifti_extension(nim, ext);
  }
  return nim;
};




::nifti_image* gourd::nifti2::create_cifti(
  const int nu,
  const int nv,
  const gourd::nifti2::intent ic,
  const int dtype
) {
  assert( nu >= 1 && "create_cifti: CIFTI dims must be > 0" );
  assert( nv >= 1 && "create_cifti: CIFTI dims must be > 0" );
  std::array<int64_t, 8> dims{ 0, 1, 1, 1, 1, nu, nv, 1 };
  dims[0] = (nv > 1) ? 6 : ((nu > 1) ? 5 : 0);
  ::nifti_image* nim = ::nifti_make_new_nim(dims.data(), dtype, 1);
  nim->intent_code = static_cast<int>(ic);
  return nim;
};





template< typename T >
inline T* gourd::nifti2::get_data_ptr(
  const ::nifti_image* const nim
) {
  // return (T*)nim->data;
  return static_cast<T*>( nim->data );
};



std::array<int64_t, 8> gourd::nifti2::dim(
  const ::nifti_image* const nim
) {
  std::array<int64_t, 8> d;
  for ( size_t i = 0; i < d.size(); i++ )  d[i] = nim->dim[i];
  return d;
};



std::vector<int> gourd::nifti2::get_dims(
  const ::nifti_image* const nim
) {
  const int maxdim = 7;  // Per nifti2_io.h
  std::vector<int> dim;
  dim.reserve( maxdim );
  for ( int i = 1; i <= maxdim; i++ ) {
    // nim->dim[0] is the number of "used" dimensions
    if ( nim->dim[i] > 1 )
      dim.push_back( nim->dim[i] );
  }
  dim.shrink_to_fit();
  return dim;
};



void gourd::nifti2::replace_cifti_extension(
  ::nifti_image* const nim,
  const std::string& ext
) {
  /* Per nifti1.h, extensions' size must be a multiple of 16.
   * There's a further 8 bit offset for 'esize' and 'ecode'
   * fields.
   *  -> Below, replacing ext.length() by utf8::distance
   *     might be safer? Not sure how always CIFTI
   *     extensions use utf-8 encoding. May need a check
   *     on the XML header line
   */
  std::string e = ext + std::string(16 - ext.length() % 16 + 8, '\0');
  /* Out with the old (extension), in with the new */
  nim->num_ext = 1;
  if ( nim->ext_list ) {
    delete[] nim->ext_list->edata;
  }
  else {
    nim->ext_list = new ::nifti1_extension;
  }
  nim->ext_list->esize = e.length() + 8;
  nim->ext_list->ecode = 32;  // <-
  nim->ext_list->edata = new char[ e.size() ];
  strncpy( nim->ext_list->edata, e.c_str(), e.size() );
};




#include "gourd/nifti_uses_datatype.inl"


#endif  // _GOURD_NIFTI_2_
