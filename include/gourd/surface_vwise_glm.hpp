
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/options.hpp"
#include "gourd/rng.hpp"
#include "gourd/data/gplm_smoothed_sstat.hpp"


#ifndef _GOURD_SURFACE_VWISE_GLM_
#define _GOURD_SURFACE_VWISE_GLM_

namespace gourd {


  template< typename T >
  class surface_vwise_glm {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;

    /*! (constructor) */
    surface_vwise_glm(
      const gourd::gplm_smoothed_sstat<T>& data
    );

    surface_vwise_glm() = default;

    mat_type beta() const;
    vector_type sigma() const;

    double log_likelihood( const gourd::gplm_smoothed_sstat<T>& data ) const;

    void update( const gourd::gplm_smoothed_sstat<T>& data );

  private:
    /* Model parameters */
    mat_type gamma_;
    vector_type sigma_sq_inv_;

    /* Auxiliary parameters */
    mat_type mu_;  /* Post. mean of gamma */
    mat_type vt_;
    vector_type d_inv_;

    void update_gamma();
    void update_sigma( const gourd::gplm_smoothed_sstat<T>& data );

    void set_initial_gamma( const gourd::gplm_smoothed_sstat<T>& data );
    void set_initial_sigma( const gourd::gplm_smoothed_sstat<T>& data );

    double mprod_trace(
      const mat_type& a,
      const mat_type& b,
      const vector_type& v
    ) const;
  };
  // class surface_vwise_glm


}  // namespace gourd




/*
 * (X' X)^-1 = (V D U' U D V')^-1 = V D^-2 V'
 */

template< typename T >
gourd::surface_vwise_glm<T>::surface_vwise_glm(
  const gourd::gplm_smoothed_sstat<T>& data
) {
  set_initial_gamma( data );
  set_initial_sigma( data );  /* Must be called after 
			       * set_initial_gamma */
};






template< typename T >
void gourd::surface_vwise_glm<T>::update(
  const gourd::gplm_smoothed_sstat<T>& data
) {
  update_sigma( data );
  update_gamma();
};


template< typename T >
void gourd::surface_vwise_glm<T>::update_gamma() {
  /* gamma = V' B 
   * var(gamma) = V' var(B) V
   *   = sigma^2 V' V D^-2 V' V
   *   = sigma^2 D^-2
   */
  for ( int i = 0; i < gamma_.rows(); i++ ) {
    const T sig = 1 / std::sqrt(sigma_sq_inv_.coeffRef(i));
    std::normal_distribution<T> normal(0, 1);
    for ( int j = 0; j < gamma_.cols(); j++ ) {
      T z = sig * d_inv_.coeffRef(j) * normal(gourd::urng());
      gamma_.coeffRef(i, j) = mu_.coeffRef(i, j) + z;
    }
  }
};


template< typename T >
void gourd::surface_vwise_glm<T>::update_sigma(
  const gourd::gplm_smoothed_sstat<T>& data
) {
  for ( int i = 0; i < sigma_sq_inv_.size(); i++ ) {
    const T shape = 0.5 * data.n() + 1;
    const T rss = data.yssq().coeffRef(i) +
      ( gamma_.row(i) * data.xsvd_d().asDiagonal() *
	(-2 * data.yu().row(i).adjoint() +
	 data.xsvd_d().asDiagonal() * gamma_.row(i).adjoint()
	 ) ).coeff(0);
    std::gamma_distribution<T> gam( shape, 2 / rss );
    sigma_sq_inv_.coeffRef(i) = gam(gourd::urng());
  }
};





template< typename T >
void gourd::surface_vwise_glm<T>::set_initial_gamma(
  const gourd::gplm_smoothed_sstat<T>& data
) {
  vt_ = data.xsvd_v().adjoint();
  d_inv_ = data.xsvd_d().cwiseInverse();
  for ( int i = 0; i < d_inv_.size(); i++ ) {
    if ( isinf(d_inv_.coeffRef(i)) )
      d_inv_.coeffRef(i) = 0;
  }
  //
  mu_ = data.yu() * d_inv_.asDiagonal();
  gamma_ = mu_;
};

template< typename T >
void gourd::surface_vwise_glm<T>::set_initial_sigma(
  const gourd::gplm_smoothed_sstat<T>& data
) {
  int df = data.n() - data.p();
  df = ( df < 1 ) ? 1 : df;
  sigma_sq_inv_.resize( data.nloc() );
  for ( int i = 0; i < data.nloc(); i++ ) {
    T eta = (data.xsvd_u() * data.yu().row(i).adjoint()).sum();
    sigma_sq_inv_.coeffRef(i) =
      df / ( data.yssq().coeffRef(i) - eta * eta / df );
    if ( isnan(sigma_sq_inv_.coeffRef(i)) ||
	 sigma_sq_inv_.coeffRef(i) <= 0 ) {
      sigma_sq_inv_.coeffRef(i) = 1;
    }
  }
};



template< typename T >
typename gourd::surface_vwise_glm<T>::mat_type
gourd::surface_vwise_glm<T>::beta() const {
  return gamma_ * vt_;
};

template< typename T >
typename gourd::surface_vwise_glm<T>::vector_type
gourd::surface_vwise_glm<T>::sigma() const {
  return sigma_sq_inv_.cwiseInverse().cwiseSqrt();
};


template< typename T >
double gourd::surface_vwise_glm<T>::log_likelihood(
  const gourd::gplm_smoothed_sstat<T>& data
) const {
  const double xterm = mprod_trace(
    gamma_, data.yu() * data.xsvd_d().asDiagonal(), sigma_sq_inv_ );
  const double qf_gamma = mprod_trace(
    gamma_, gamma_ * data.xsvd_d().cwiseAbs2().asDiagonal(),
    sigma_sq_inv_ );
  const double qf_y = static_cast<double>(
    (data.yssq().adjoint() * sigma_sq_inv_).coeff(0) );
  return -0.5 * qf_y + xterm + -0.5 * qf_gamma;
};



/* Compute tr(a' diag(v) b) */
template< typename T >
double gourd::surface_vwise_glm<T>::mprod_trace(
  const typename gourd::surface_vwise_glm<T>::mat_type& a,
  const typename gourd::surface_vwise_glm<T>::mat_type& b,
  const typename gourd::surface_vwise_glm<T>::vector_type& v
) const {
  assert( a.cols() == b.cols() &&
	  a.rows() == b.rows() &&
	  a.rows() == v.size() &&
	  "surface_vwise_gm:matrix trace: dimensions must agree" );
  double trace = 0;
  for ( int j = 0; j < a.cols(); j++ )
    trace += static_cast<double>(
      (a.col(j).adjoint() * v.asDiagonal() * b.col(j)).coeff(0) );
  return trace;
};


#endif  // _GOURD_SURFACE_VWISE_GLM_
