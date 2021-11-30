

#include "gourd/surface_gplmix_model.hpp"
#include "gourd/data/gplm_full_data.hpp"


#ifndef _GOURD_LOOCV_
#define _GOURD_LOOCV_


namespace gourd {

  /*! Compute Leave-one-out Cross-Validation Error 
   */
  template< typename T > 
  double loocv(
    const gourd::gplm_full_data<T>& data,
    const gourd::surface_gplmix_model<T>& model
  ) {
    using mat_type =
      typename gourd::surface_gplmix_model<T>::mat_type;
    // Compute hat matrix diagonal terms
    const mat_type dhat = model.smdiag( data );
    // Compute LOOCV
    double val = 0;
    double a, b, err = 0;
    for ( int i = 0; i < data.n(); i++ ) {
      for ( int s = 0; s < data.nloc(); s++ ) {
	double res = data.y(i, s) -
	  ( data.x().row(i) * model.beta().row(s).adjoint() );
	double fac = 1 - dhat.coeffRef(s, i);
	fac = (std::abs(fac) < 1e-8) ? 1e-8 : fac;
	double tmp = res / fac;
	a = tmp * tmp - err;
	b = val + a;
	err = (b - val) - a;
	val = b;
      }
    }
    return val / dhat.size();
  };


} // namespace abseil


#endif  // _GOURD_LOOCV_


