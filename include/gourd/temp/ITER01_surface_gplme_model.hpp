
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

// #include <omp.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/gplm_sstat.hpp"
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
  /*! Gaussian process linear model for cortical surface data 
   *
   * 
   */
  template< typename T >
  class surface_gplme_model {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;

    /*
     * Needs to construct:
     *  - nnp_hess( vector<cartesian_coordinate>,
     *              covariance_functor, T radius, dist_code )
     *  - learning_rate( double eps0, double target_mh )
     */
    /*! (constructor) */
    template< size_t D >
    surface_gplme_model(
      const gourd::gplm_sstat<T>& data,
      const abseil::covariance_functor<T, D>* const cov,
      const T nngp_radius,
      const gourd::dist_code distance,
      const int integrator_steps = 10,  /*!< HMC integrator steps */
      const double eps0 = 0.1,     /*!< HMC initial step size */
      const double mass_rad = 4,   /*!< HMC mass matrix n'hood radius */
      const double alpha0 = 0.65,  /*!< HMC target Metropolis-Hastings rate */
      const double eps_min = 1e-5, /*!< HMC minimum step size */
      const double g0 = 0.05,      /*!< HMC dual-averaging tuning parameter */
      const double t0 = 10,        /*!< HMC dual-averaging tuning parameter */
      const double k0 = 0.75       /*!< HMC dual-averaging tuning parameter */
    );

    surface_gplme_model() = default;
    
    mat_type beta() const;
    // vector_type sigma() const;
    T tau() const;
    T xi() const;
    
    double update(
      const gourd::gplm_sstat<T>& data,
      const bool update_learning_rate
    );

    bool tune_initial_stepsize(
      const gourd::gplm_sstat<T>& data,
      const int maxit = 100
    );

    void warmup(
      const gourd::gplm_sstat<T>& data,
      const int niter
    );

    void profile(
      const gourd::gplm_sstat<T>& data,
      const int nrep = 10
    );
    
    
  private:

    /* Model parameters */
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type gamma_star_;
    T tau_sq_inv_;
    gourd::nnp_hess<T> v_inv_;  /* Residual variance (not updated) */
    gourd::nnp_hess<T> c_inv_;
    // vector_type sigma_sq_inv_;  //
    // T xi_;

    /* Parameters related to HMC */
    mat_type momentum_;
    mat_type vt_;                /* X = U D ~Vt~ */
    double energy_initial_;      /* Partial log posterior */
    double energy_momentum_;
    double energy_proposal_;
    double log_prior_kernel_gamma_;  /* Prior on gamma_: log kernel */
    //
    int leapfrog_steps_;
    abseil::learning_rate lr_;  //
    gourd::nnp_hess<T> mass_;


    double partial_loglik(
      const gourd::gplm_sstat<T>& data,
      const mat_type& g
    ) const;

    double partial_logprior(
      const mat_type& g
    ) const;

    /* Compute trace of [a' diag(v) b] */
    double mprod_trace(
      const mat_type& a,
      const mat_type& b,
      const vector_type& v
    ) const;

    mat_type grad_log_prior( const mat_type& g ) const;
    mat_type grad_log_likelihood(
      const gourd::gplm_sstat<T>& data,
      const mat_type& g
    ) const;

    mat_type grad_g(
      const gourd::gplm_sstat<T>& data,
      const mat_type& g
    );
    

    double update_gamma_hmc(
      const gourd::gplm_sstat<T>& data,
      const int integrator_steps,
      const bool update = true
    );
    
    // void update_sigma_xi( const gourd::gplm_sstat<T>& data );
    
    void update_tau();

    double potential_energy() const;
    
    void sample_momentum_and_energy();

    void set_initial_gamma(
      const gourd::gplm_sstat<T>& data
    );

    // void set_initial_sigma(
    //   const gourd::gplm_sstat<T>& data
    // );

    void set_initial_values(
      const gourd::gplm_sstat<T>& data
    );
    
  };
  // class surface_gplme_model
  /* ****************************************************************/
  
};
// namespace gourd



template< typename T >
double gourd::surface_gplme_model<T>::partial_loglik(
  const gourd::gplm_sstat<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  const mat_type v_inv_g = v_inv_.rmul( g );
  double cross_term = 0;
  double quad_term = 0;
  for ( int j = 0; j < g.cols(); j++ ) {
    double dsq = static_cast<double>( data.xsvd_d().coeffRef(j) );
    dsq *= dsq;
    cross_term += dsq *
      (v_inv_g.col(j).adjoint() * data.yu().col(j)).coeff(0);
    //
    quad_term += dsq * (v_inv_g.col(j).adjoint() * g.col(j)).coeff(0);
  }
  return -0.5 * quad_term + cross_term;
};



template< typename T >
double gourd::surface_gplme_model<T>::partial_logprior(
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return -0.5 * tau_sq_inv_ * c_inv_.trqf(g);
};


/* Compute tr(a' diag(v) b) */
template< typename T >
double gourd::surface_gplme_model<T>::mprod_trace(
  const typename gourd::surface_gplme_model<T>::mat_type& a,
  const typename gourd::surface_gplme_model<T>::mat_type& b,
  const typename gourd::surface_gplme_model<T>::vector_type& v
) const {
  assert( a.cols() == b.cols() &&
	  a.rows() == b.rows() &&
	  a.rows() == v.size() &&
	  "surface_gplme_model:matrix trace: dimensions must agree" );
  double trace = 0;
  for ( int j = 0; j < a.cols(); j++ )
    trace += static_cast<double>(
      (a.col(j).adjoint() * v.asDiagonal() * b.col(j)).coeff(0) );
  return trace;
};



template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_log_prior(
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return -tau_sq_inv_ * c_inv_.rmul( g );
};


template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_log_likelihood(
  const gourd::gplm_sstat<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return v_inv_.rmul(
      ( data.yu() - g * data.xsvd_d().asDiagonal() ) *
      data.xsvd_d().asDiagonal()
    );
};



template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_g(
  const gourd::gplm_sstat<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) {
  return grad_log_likelihood(data, g) + grad_log_prior(g);
};





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplme_model<T>::update(
  const gourd::gplm_sstat<T>& data,
  const bool update_learning_rate
) {
  update_tau();
  // update_sigma_xi( data );
  const double alpha = update_gamma_hmc( data, leapfrog_steps_ );
  if ( update_learning_rate )
    lr_.adapt( alpha );
  std::cout << "\t\t<< \u03b1 = " << alpha
	    << "; \u03c4 = " << std::sqrt(1 / tau_sq_inv_)
	    << " >>"
	    << std::endl;
  return alpha;
};

// \alpha <==> \u03b1
// \tau   <==> \u03c4




template< typename T > 
double gourd::surface_gplme_model<T>::update_gamma_hmc(
  const gourd::gplm_sstat<T>& data,
  const int integrator_steps,
  const bool update
) {
  std::uniform_real_distribution<double> unif(0, 1);
  const T eps =
    lr_.eps( (update ? 0.9 + unif(gourd::urng()) * 0.2 : 1) );
  double log_prior_star, alpha;
  T k = 0.5;
  sample_momentum_and_energy();
  energy_initial_ = -partial_loglik(data, gamma_) -
    partial_logprior(gamma_);
  gamma_star_ = gamma_;
  momentum_ += k * eps * grad_g( data, gamma_ );
  for ( int step = 0; step < integrator_steps; step++ ) {
    k = (step == (integrator_steps - 1)) ? 0.5 : 1;
    gamma_star_.noalias() += eps * mass_.irmul( momentum_ );
    momentum_.noalias() += k * eps * grad_g( data, gamma_star_ );
  }
  // ( momentum_ *= -1 )
  log_prior_star = partial_logprior(gamma_star_);
  energy_proposal_ = -partial_loglik(data, gamma_star_) -
    log_prior_star;
  alpha = std::exp( -energy_proposal_ - potential_energy() +
		    energy_initial_ + energy_momentum_ );
  alpha = isnan(alpha) ? 0 : alpha;
  if ( update  &&  unif(gourd::urng()) < alpha ) {
    gamma_ = gamma_star_;
    log_prior_kernel_gamma_ = log_prior_star;
  }
  return (alpha > 1) ? 1 : alpha;
};
  // \epsilon <---> \u03b5







template< typename T > 
void gourd::surface_gplme_model<T>::update_tau() {
  const int s = gamma_.rows();
  const int p = gamma_.cols();
  const T shape = 0.5 * s * p + 1;
  const T rate = 0.5 * -log_prior_kernel_gamma_ / tau_sq_inv_ + 1;
  std::gamma_distribution<T> gam( shape, 1 / rate );
  tau_sq_inv_ = gam(gourd::urng());
};
/* ****************************************************************/




template< typename T > 
void gourd::surface_gplme_model<T>::sample_momentum_and_energy() {
  energy_momentum_ = 0;
  /* Profile outer loop as parallel */
#pragma omp parallel for reduction(+:energy_momentum_) shared(momentum_)
  for ( int j = 0; j < momentum_.cols(); j++ ) {
    std::normal_distribution<double> normal(0, 1);
    double z;        // Random normal draw
    double a, b, err = 0;
    double partial_sum = 0;
    for ( int i = 0; i < momentum_.rows(); i++ ) {
      z = normal(gourd::urng());
      a = z * z - err;
      b = partial_sum + a;
      err = (b - partial_sum) - a;
      partial_sum = b;
      momentum_.coeffRef(i, j) = static_cast<T>( z );
    }
    /* Critical section */
    energy_momentum_ += partial_sum;
  }
  energy_momentum_ *= 0.5;
  momentum_ = mass_.hprod(momentum_).eval();
  /* ^^ Inefficient? Not really. Appears to be very marginally faster 
   * than preallocating the memory and copying into momentum_
   */
};



template< typename T > 
double gourd::surface_gplme_model<T>::potential_energy() const {
  return 0.5 * mass_.triqf(momentum_);
};






/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplme_model<T>::set_initial_values(
  const gourd::gplm_sstat<T>& data
) {
  tau_sq_inv_ = 1;
  // set_initial_sigma( data );
  set_initial_gamma( data );
  update_tau(); /* Must be called AFTER set_initial_gamma() */
  log_prior_kernel_gamma_ *= tau_sq_inv_;
  //
  std::cout << "Initial gamma:\n"
	    << gamma_.topRows(10)
	    << "\n" << std::endl;
};




template< typename T > 
bool gourd::surface_gplme_model<T>::tune_initial_stepsize(
  const gourd::gplm_sstat<T>& data,
  const int maxit
) {
  bool tuning_needed = true;
  int it = 0;
  double alpha;
  while ( it < maxit && tuning_needed ) {
    alpha = update_gamma_hmc( data, 1, false );
    tuning_needed = lr_.adjust_initial_value( alpha );
    it++;
  }
  if ( it >= maxit && tuning_needed ) {
    std::cerr << "surface_gplme_model: initial HMC step size "
	      << "not found after " << it << " iterations\n"
	      << "\t(Final value was: " << lr_ << ")\n";
  }
  else {
    std::cerr << "surface_gplme_model: HMC step size "
	      << "tuning took " << it << " iterations\n"
	      << "\tValue: " << lr_
	      << std::endl;
  }
  return !tuning_needed;
};




template< typename T > 
void gourd::surface_gplme_model<T>::warmup(
  const gourd::gplm_sstat<T>& data,
  const int niter
) {
  assert(niter >= 0 &&
	 "surface_gplme_model: negative warmup iterations");
  const bool eps_tuned = tune_initial_stepsize(data, niter/2);
  if ( !eps_tuned && niter/2 > 0 ) {
    std::cerr << "\t*** HMC step size tuning failed\n";
  }
  for ( int i = 0; i < niter; i++ )
    update(data, true);
  lr_.fix();
};





template< typename T > 
void gourd::surface_gplme_model<T>::profile(
  const gourd::gplm_sstat<T>& data,
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
    update_gamma_hmc( data, 10 );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\bbeta(\\cdot)$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";


  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    grad_g( data, gamma_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }  
  std::cout << "Evaluate gradient  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    partial_loglik( data, gamma_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Evaluate log-likelihood  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    partial_logprior( gamma_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Evaluate log-prior  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  std::cout << "\\hline\n";
  

  
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
  

  std::cout << "\\hline\n";
  std::cout << "\\end{tabular}\n"
	    << "\\end{table}\n";
  
  std::cout << std::endl;
};





template< typename T > 
void gourd::surface_gplme_model<T>::set_initial_gamma(
  const gourd::gplm_sstat<T>& data
) {
  const vector_type di = data.xsvd_d().cwiseInverse();
  gamma_ = data.yu() * di.asDiagonal();
  vt_ = data.xsvd_v().adjoint();
  /* Jitter initial values */
  // const Eigen::LDLT<mat_type> vt_decomp = vt_.ldlt();
  T tau = std::sqrt(
    (gamma_.cwiseAbs2().colwise().mean() -
     gamma_.colwise().mean().cwiseAbs2()).maxCoeff()
   );
  std::normal_distribution<T> normal(0, 0.1 * tau);
  vector_type z( gamma_.cols() );
  for ( int i = 0; i < gamma_.rows(); i++ ) {
    for ( int j = 0; j < z.size(); j++ ) {
      z.coeffRef(j) = normal(gourd::urng());
    }
    // gamma_.row(i) += vt_decomp.solve( di.asDiagonal() * z ).adjoint();
    gamma_.row(i) += ( data.xsvd_v() * di.asDiagonal() * z ).adjoint();
  }
  /* Set related initial values */
  gamma_star_.resize( gamma_.rows(), gamma_.cols() );
  momentum_.resize( gamma_.rows(), gamma_.cols() );
  log_prior_kernel_gamma_ = partial_logprior(gamma_);
  // energy_initial_ = partial_loglik( data, gamma_ ) +
  //   log_prior_kernel_gamma_;
};



/* ****************************************************************/


/* E x^2 - (E x)^2 
 * E Y = Y U U'
 */


/* ****************************************************************/
/* 
 *                               Getters
 */
template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::beta() const {
  return gamma_ * vt_;
};



template< typename T > inline
T gourd::surface_gplme_model<T>::tau() const {
  return 1 / std::sqrt( tau_sq_inv_ );
};

/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplme_model<T>::surface_gplme_model(
  const gourd::gplm_sstat<T>& data,
  const abseil::covariance_functor<T, D>* const cov,
  const T nngp_radius,
  const gourd::dist_code distance,
  const int integrator_steps,
  const double eps0,
  const double mass_rad,
  const double alpha0,
  const double eps_min,
  const double g0,
  const double t0,
  const double k0
) {
  assert(integrator_steps > 0 &&
	 "surface_gplme_model: negative HMC integrator steps");
  assert(nngp_radius >= 0 &&
	 "surface_gplme_model: non-positive NNGP neighborhood radius");
  assert(mass_rad >= 0 &&
	 "surface_gplme_model: non-positive mass matrix neighborhood");
  //
  /* Set residual precision matrix */
  v_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov,    // <- should be different from below
    nugget,
    nngp_radius,
    distance,
    false
  );
  /* Set Hessian */
  c_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov,
    nngp_radius,
    distance
  );
  /* Set Mass matrix */
  mass_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov,
    mass_rad,
    distance
  );
  /* Set HMC learning rate/step size */
  lr_ = abseil::learning_rate( eps0, alpha0, eps_min, g0, t0, k0 );
  leapfrog_steps_ = integrator_steps;
  /* Initialize parameters */
  set_initial_values( data );
};


#endif  // _GOURD_SURFACE_GPLME_MODEL_
