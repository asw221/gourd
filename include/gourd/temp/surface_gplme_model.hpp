
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

// #include <omp.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/gplm_full_data.hpp"
#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/rng.hpp"

// Profiling
#include "abseil/timer.hpp"
//


#ifndef _GOURD_SURFACE_GPLME_MODEL_
#define _GOURD_SURFACE_GPLME_MODEL_

/* To do:
 *  - Credible set estimation
 *  - Enable output CIFTI files (*.dtseries.nii format)
 */


namespace gourd {

  /* ****************************************************************/
  /*! Gaussian process linear mixed effect model for cortical 
   *  surface data 
   * 
   */
  template< typename T >
  class surface_gplme_model {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;
    typedef typename Eigen::SparseMatrix<T> spmat_type;

    /*
     * Needs to construct:
     *  - nnp_hess( vector<cartesian_coordinate>,
     *              covariance_functor, T radius, dist_code )
     *  - learning_rate( double eps0, double target_mh )
     */
    /*! (constructor) */
    template< size_t D >
    surface_gplme_model(
      const gourd::gplm_full_data<T>& data,
      const abseil::covariance_functor<T, D>* const cov,
      const T nngp_radius,
      const gourd::dist_code distance
    );

    surface_gplme_model() = default;

    const mat_type& omega() const;
    mat_type beta() const;
    vector_type sigma() const;

    double lp() const;
    double tau() const;
    double xi() const;
    double zeta() const;
    
    double update(
      const gourd::gplm_full_data<T>& data
    );
    double gibbs_update(
      const gourd::gplm_full_data<T>& data
    );

    void profile(
      const gourd::gplm_full_data<T>& data,
      const int nrep = 10
    );
    
    
  private:

    /* Model parameters */
    double tau_sq_inv_;
    double xi_;
    double zeta_sq_inv_;
    
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type omega_;
    
    vector_type sigma_sq_inv_;
    
    spmat_type c_inv_;
    spmat_type var_temp_;

    mat_type vt_;
    mat_type dut_;
    Eigen::SimplicialLDLT<spmat_type> vldlt_;

    /* Log posterior */
    double dlp_;
    double logpost_;
    double prev_logpost_;

    

    void update_gamma(
      const gourd::gplm_full_data<T>& data
    );
    void update_omega(
      const gourd::gplm_full_data<T>& data
    );

    /* Update variance terms
     * Each update_<var> also updates logpost_ for relevant terms
     */
    void update_sigma_xi(
      const gourd::gplm_full_data<T>& data
    );
    void update_tau();
    void update_zeta();

    
    void gibbs_update_gamma(
      const gourd::gplm_full_data<T>& data
    );
    void gibbs_update_omega(
      const gourd::gplm_full_data<T>& data
    );

    /* Update variance terms
     * Each update_<var> also updates logpost_ for relevant terms
     */
    void gibbs_update_sigma_xi(
      const gourd::gplm_full_data<T>& data
    );
    void gibbs_update_tau();
    void gibbs_update_zeta();
    

    void set_initial_values(
      const gourd::gplm_full_data<T>& data
    );
    
    void set_initial_gamma(
      const gourd::gplm_full_data<T>& data
    );
    void set_initial_sigma(
      const gourd::gplm_full_data<T>& data
    );
    
  };
  // class surface_gplme_model
  /* ****************************************************************/
  
};
// namespace gourd





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplme_model<T>::update(
  const gourd::gplm_full_data<T>& data
) {
  logpost_ = 0;
  update_omega( data );
  update_gamma( data );
  /* Update variance components last: these functions
   * also add to logpost_ to update the log-posterior
   */
  update_sigma_xi( data );
  update_tau();
  update_zeta();
  //
  dlp_ = logpost_ - prev_logpost_;
  prev_logpost_ = logpost_;
  return dlp_;
};




template< typename T >
void gourd::surface_gplme_model<T>::update_omega(
  const gourd::gplm_full_data<T>& data
) {
  var_temp_ = zeta_sq_inv_ * c_inv_;
  for ( int i = 0; i < var_temp_.cols(); i++ ) {
    var_temp_.coeffRef(i, i) += sigma_sq_inv_.coeffRef(i);
  }
  vldlt_.factorize( var_temp_ );
  omega_ = vldlt_.solve(
    sigma_sq_inv_.asDiagonal() * (data.y() - gamma_ * dut_)
  );
};


template< typename T >
void gourd::surface_gplme_model<T>::update_gamma(
  const gourd::gplm_full_data<T>& data
) {
  const mat_type eta = sigma_sq_inv_.asDiagonal() *
    ( (data.y() - omega_) * dut_.adjoint() );
  /* Decompositions must be compued block-wise */
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    T dsq = data.xsvd_d().coeffRef(j) * data.xsvd_d().coeffRef(j);
    var_temp_ = static_cast<T>( tau_sq_inv_ ) * c_inv_;
    for ( int i = 0; i < var_temp_.cols(); i++ ) {
      var_temp_.coeffRef(i, i) += dsq * sigma_sq_inv_.coeffRef(i);
    }
    vldlt_.factorize( var_temp_ );
    gamma_.col(j) = vldlt_.solve( eta.col(j) );
  }
};


template< typename T > 
void gourd::surface_gplme_model<T>::update_sigma_xi(
  const gourd::gplm_full_data<T>& data
) {
  const int nloc = data.nloc();
  const T shape = 0.5 * data.n() + 0.5;
  // const vector_type dsq = data.xsvd_d().cwiseAbs2();
  double sum_isig = 0;
  for ( int i = 0; i < nloc; i++ ) {
    T rss = ( data.y().row(i) - gamma_.row(i) * dut_ -
	      omega_.row(i) ).cwiseAbs2().sum();
    T rate = 0.5 * rss + static_cast<T>( xi_ );
    //
    /* update sigma_sq_inv_(i) */
    sigma_sq_inv_.coeffRef(i) = (shape - 1) / rate;
    //
    double isig =
      static_cast<double>( sigma_sq_inv_.coeffRef(i) );
    double ln_isig = std::log( isig );
    sum_isig += isig;
    //
    /* log likelihood */
    logpost_ += -0.5 * isig * rss + 0.5 * data.n() * ln_isig;
    /* partial log prior of sigma */
    logpost_ += -0.5 * ln_isig;
  }
  /* Update prior on the rate of the sigma's */
  xi_ = (0.5 * nloc - 0.5) / (1 + sum_isig);
  //
  /* other part of log prior on sigma's */
  logpost_ += -xi_ * sum_isig;
  /* log prior xi */
  logpost_ += -0.5 * std::log( xi_ ) - xi_;
};



template< typename T > 
void gourd::surface_gplme_model<T>::update_tau() {
  const int s = gamma_.rows();
  const int p = gamma_.cols();
  const double shape = 0.5 * s * p + 1;
  double cross_term = 0;
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    cross_term += static_cast<double>(
      ( gamma_.col(j).adjoint() * (c_inv_ * gamma_.col(j)) )
      .coeff(0) );
  }
  double rate = 0.5 * cross_term + 1;
  tau_sq_inv_ = (shape - 1) / rate;
  //
  /* log prior gamma */
  logpost_ += 0.5*s*p * std::log(tau_sq_inv_) +
    -0.5 * tau_sq_inv_ * cross_term;
  /* log prior tau */
  logpost_ += -tau_sq_inv_;
};



template< typename T > 
void gourd::surface_gplme_model<T>::update_zeta() {
  const int s = omega_.rows();
  const int n = omega_.cols();
  const double shape = 0.5 * s * n + 1;
  double cross_term = 0;
  for ( int j = 0; j < omega_.cols(); j++ ) {
    cross_term += static_cast<double>(
      ( omega_.col(j).adjoint() * (c_inv_ * omega_.col(j)) )
      .coeff(0) );
  }
  double rate = 0.5 * cross_term + 1;
  zeta_sq_inv_ = (shape - 1) / rate;
  //
  /* log prior omega */
  logpost_ += 0.5*s*n * std::log(zeta_sq_inv_) +
    -0.5 * zeta_sq_inv_ * cross_term;
  /* log prior zeta */
  logpost_ += -zeta_sq_inv_;
};



/* ***************************************************************
 *                       Gibbs updates 
 * ***************************************************************/

template< typename T > 
double gourd::surface_gplme_model<T>::gibbs_update(
  const gourd::gplm_full_data<T>& data
) {
  logpost_ = 0;
  gibbs_update_omega( data );
  gibbs_update_gamma( data );
  /* Update variance components last: these functions
   * also add to logpost_ to update the log-posterior
   */
  gibbs_update_sigma_xi( data );
  gibbs_update_tau();
  gibbs_update_zeta();
  //
  dlp_ = logpost_ - prev_logpost_;
  prev_logpost_ = logpost_;
  return dlp_;
};



template< typename T >
void gourd::surface_gplme_model<T>::gibbs_update_omega(
  const gourd::gplm_full_data<T>& data
) {
  var_temp_ = zeta_sq_inv_ * c_inv_;
  for ( int i = 0; i < var_temp_.cols(); i++ ) {
    var_temp_.coeffRef(i, i) += sigma_sq_inv_.coeffRef(i);
  }
  vldlt_.factorize( var_temp_ );
  // Set omega to mean
  omega_ = vldlt_.solve(
    sigma_sq_inv_.asDiagonal() * (data.y() - gamma_ * dut_)
  );
  // Add sampling noise
  for ( int j = 0; j < omega_.cols(); j++ ) {
    std::normal_distribution<T> normal(0, 1);
    vector_type z(omega_.rows());
    for ( int i = 0; i < z.size(); i++ ) {
      z.coeffRef(i) = normal(gourd::urng());
    }
    omega_.col(j).noalias() += vldlt_.solve( vldlt_.matrixL() *
      (vldlt_.vectorD().cwiseSqrt().asDiagonal() * z) );
  }
};


template< typename T >
void gourd::surface_gplme_model<T>::gibbs_update_gamma(
  const gourd::gplm_full_data<T>& data
) {
  const mat_type eta = sigma_sq_inv_.asDiagonal() *
    ( (data.y() - omega_) * dut_.adjoint() );
  /* Decompositions must be compued block-wise */
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    T dsq = data.xsvd_d().coeffRef(j) * data.xsvd_d().coeffRef(j);
    var_temp_ = static_cast<T>( tau_sq_inv_ ) * c_inv_;
    for ( int i = 0; i < var_temp_.cols(); i++ ) {
      var_temp_.coeffRef(i, i) += dsq * sigma_sq_inv_.coeffRef(i);
    }
    vldlt_.factorize( var_temp_ );
    // Set gamma_j to mean
    gamma_.col(j) = vldlt_.solve( eta.col(j) );
    // Add sampling noise
    std::normal_distribution<T> normal(0, 1);
    vector_type z(gamma_.rows());
    for ( int i = 0; i < z.size(); i++ ) {
      z.coeffRef(i) = normal(gourd::urng());
    }
    gamma_.col(j).noalias() += vldlt_.solve( vldlt_.matrixL() *
      (vldlt_.vectorD().cwiseSqrt().asDiagonal() * z) );
  }
};

/*
Vi = LDU
Want: V^1/2 z
  ==> solve: Vi x = L D^1/2 z 
  Since V = U^-1 D^-1 L^-1
  ==> x = U^-1 D^-1/2 z
  ==> var(x) = U^-1 D^-1 L^-1 = V .
 */




template< typename T > 
void gourd::surface_gplme_model<T>::gibbs_update_sigma_xi(
  const gourd::gplm_full_data<T>& data
) {
  const int nloc = data.nloc();
  const T shape = 0.5 * data.n() + 0.5;
  // const vector_type dsq = data.xsvd_d().cwiseAbs2();
  double sum_isig = 0;
  for ( int i = 0; i < nloc; i++ ) {
    T rss = ( data.y().row(i) - gamma_.row(i) * dut_ -
	      omega_.row(i) ).cwiseAbs2().sum();
    T rate = 0.5 * rss + static_cast<T>( xi_ );
    //
    /* update sigma_sq_inv_(i) */
    std::gamma_distribution<T> gams(shape, 1 / rate);
    sigma_sq_inv_.coeffRef(i) = gams(gourd::urng());
    //
    double isig =
      static_cast<double>( sigma_sq_inv_.coeffRef(i) );
    double ln_isig = std::log( isig );
    sum_isig += isig;
    //
    /* log likelihood */
    logpost_ += -0.5 * isig * rss + 0.5 * data.n() * ln_isig;
    /* partial log prior of sigma */
    logpost_ += -0.5 * ln_isig;
  }
  /* Update prior on the rate of the sigma's */
  std::gamma_distribution<double>
    gamx(0.5 * nloc + 0.5, 1 / (1 + sum_isig));
  xi_ = gamx(gourd::urng());
  //
  /* other part of log prior on sigma's */
  logpost_ += -xi_ * sum_isig;
  /* log prior xi */
  logpost_ += -0.5 * std::log( xi_ ) - xi_;
};



template< typename T > 
void gourd::surface_gplme_model<T>::gibbs_update_tau() {
  const int s = gamma_.rows();
  const int p = gamma_.cols();
  const double shape = 0.5 * s * p + 1;
  double cross_term = 0;
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    cross_term += static_cast<double>(
      ( gamma_.col(j).adjoint() * (c_inv_ * gamma_.col(j)) )
      .coeff(0) );
  }
  double rate = 0.5 * cross_term + 1;
  std::gamma_distribution<double> gam(shape, 1 / rate);
  tau_sq_inv_ = gam(gourd::urng());
  //
  /* log prior gamma */
  logpost_ += 0.5*s*p * std::log(tau_sq_inv_) +
    -0.5 * tau_sq_inv_ * cross_term;
  /* log prior tau */
  logpost_ += -tau_sq_inv_;
};



template< typename T > 
void gourd::surface_gplme_model<T>::gibbs_update_zeta() {
  const int s = omega_.rows();
  const int n = omega_.cols();
  const double shape = 0.5 * s * n + 1;
  double cross_term = 0;
  for ( int j = 0; j < omega_.cols(); j++ ) {
    cross_term += static_cast<double>(
      ( omega_.col(j).adjoint() * (c_inv_ * omega_.col(j)) )
      .coeff(0) );
  }
  double rate = 0.5 * cross_term + 1;
  std::gamma_distribution<double> gam(shape, 1 / rate);
  zeta_sq_inv_ = gam(gourd::urng());
  //
  /* log prior omega */
  logpost_ += 0.5*s*n * std::log(zeta_sq_inv_) +
    -0.5 * zeta_sq_inv_ * cross_term;
  /* log prior zeta */
  logpost_ += -zeta_sq_inv_;
};

/* ****************************************************************/







/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplme_model<T>::set_initial_values(
  const gourd::gplm_full_data<T>& data
) {
  // Store parts of the decomposition of x locally
  vt_ = data.xsvd_v().adjoint();
  dut_ = data.xsvd_d().asDiagonal() * data.xsvd_u().adjoint();
  //
  tau_sq_inv_ = 1;
  xi_ = 1;
  zeta_sq_inv_ = 1;
  set_initial_gamma( data );
  set_initial_sigma( data );
  omega_ = mat_type::Zero( data.nloc(), data.n() );
  //
  /* Perform initial update to start from valid logpost, etc.
   */
  update( data );
};





template< typename T > 
void gourd::surface_gplme_model<T>::set_initial_gamma(
  const gourd::gplm_full_data<T>& data
) {
  // Mass univariate estimates
  const vector_type di = data.xsvd_d().cwiseInverse();
  gamma_ = data.y() * (data.xsvd_u() * di.asDiagonal());
};


template< typename T >
void gourd::surface_gplme_model<T>::set_initial_sigma(
  const gourd::gplm_full_data<T>& data
) {
  const int nloc = data.nloc();
  const T shape = 0.5 * data.n() + 0.5;
  T rss;  // Residual sum of squares
  sigma_sq_inv_.resize( nloc );
  for ( int i = 0; i < nloc; i++ ) {
    rss = (data.y().row(i) - gamma_.row(i) * dut_)
      .cwiseAbs2().sum();
    //
    sigma_sq_inv_.coeffRef(i) = (shape - 1) / (0.5 * rss + 1);
  }
};



/* ****************************************************************/
/*
 *                             Profile
 */
template< typename T > 
void gourd::surface_gplme_model<T>::profile(
  const gourd::gplm_full_data<T>& data,
  const int nrep
) {
  double dt;
  
  std::cout << "Profiling computations:\n";
  std::cout << "\\begin{table}\n"
	    << "\\centering\n"
	    << "\\begin{tabular}{ l c }\n"
	    << "Operation  &  Time (ms) \\\\\n";

  std::cout << "\\hline\n";

  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    update_gamma( data );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\bbeta(\\cdot)$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";


  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    update_omega( data );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\bomega(\\cdot)$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";
  

  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    update_sigma_xi( data );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\sigma^{-2}(\\cdot)$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    update_tau();
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\tau^{-2}$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    update_zeta();
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\zeta^{-2}$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";
  

  std::cout << "\\hline\n";
  std::cout << "\\end{tabular}\n"
	    << "\\end{table}\n";
  
  std::cout << std::endl;
};




/* ****************************************************************/




/* ****************************************************************/
/* 
 *                               Getters
 */
template< typename T > inline
const typename gourd::surface_gplme_model<T>::mat_type&
gourd::surface_gplme_model<T>::omega() const {
  return omega_;
};


template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::beta() const {
  return gamma_ * vt_;
};


template< typename T > inline
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::sigma() const {
  return sigma_sq_inv_.cwiseInverse().cwiseSqrt();
};



template< typename T > inline
double gourd::surface_gplme_model<T>::lp() const {
  return logpost_;
};


template< typename T > inline
double gourd::surface_gplme_model<T>::tau() const {
  return 1 / std::sqrt( tau_sq_inv_ );
};


template< typename T > inline
double gourd::surface_gplme_model<T>::xi() const {
  return xi_;
};


template< typename T > inline
double gourd::surface_gplme_model<T>::zeta() const {
  return 1 / std::sqrt( zeta_sq_inv_ );
};

/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplme_model<T>::surface_gplme_model(
  const gourd::gplm_full_data<T>& data,
  const abseil::covariance_functor<T, D>* const cov,
  const T nngp_radius,
  const gourd::dist_code distance
) {
  assert(nngp_radius >= 0 &&
	 "surface_gplme_model: non-positive NNGP "
	 "neighborhood radius");
  assert( data.n() > 1 &&
	  "surface_gplme_model: model not suitable for n <= 1" );
  //
  /* Set Hessian */
  gourd::nnp_hess<T> c_proto_(
    data.coordinates(),
    cov,
    nngp_radius,
    distance
  );
  c_inv_ = c_proto_.hessian();
  vldlt_.analyzePattern( c_inv_ );  /* <- All decompositions 
				       * are of matrices with this 
				       * same sparsity pattern */
  //
  /* Initialize parameters */
  set_initial_values( data );
};


#endif  // _GOURD_SURFACE_GPLME_MODEL_
