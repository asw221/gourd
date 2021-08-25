
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

// #include <omp.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/gplm_outcome_data.hpp"
#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/rng.hpp"

// Profiling
#include "abseil/timer.hpp"
//


#ifndef _GOURD_SURFACE_GPLCOV_MODEL_
#define _GOURD_SURFACE_GPLCOV_MODEL_

/* To do:
 *  - 
 */


namespace gourd {

  double marginal_loglik( const long n, const double* x, void* data );
  

  /* ****************************************************************/
  /*! Gaussian process linear model for cortical surface data 
   *
   * 
   */
  template< typename T >
  class surface_gplcov_model {
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
    surface_gplcov_model(
      const gourd::gplm_outcome_data<T>& data,
      abseil::covariance_functor<T, D>* const cov,
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

    surface_gplcov_model() = default;
    
    const mat_type& omega() const;
    vector_type sigma() const;
    T tau() const;
    T xi() const;
    
    double update(
      const gourd::gplm_outcome_data<T>& data,
      const bool update_learning_rate
    );

    bool tune_initial_stepsize(
      const gourd::gplm_outcome_data<T>& data,
      const int maxit = 100
    );

    void warmup(
      const gourd::gplm_outcome_data<T>& data,
      const int niter
    );

    void profile(
      const gourd::gplm_outcome_data<T>& data,
      const int nrep = 10
    );
    
    
  private:

    /* Model parameters */
    mat_type omega_;            /* omega = spatial intercepts */
    mat_type omega_star_;
    vector_type sigma_sq_inv_;  //
    T tau_sq_inv_;
    T xi_;
    gourd::nnp_hess<T> c_inv_;

    /* Parameters related to HMC */
    mat_type momentum_;
    double energy_initial_;      /* Partial log posterior */
    double energy_momentum_;
    double energy_proposal_;
    double log_prior_kernel_omega_;  /* Prior on omega_: log kernel */
    //
    int leapfrog_steps_;
    abseil::learning_rate lr_;  //
    gourd::nnp_hess<T> mass_;
    //
    abseil::covariance_functor<T, 3>* cov_ptr_;
    //


    double partial_loglik(
      const gourd::gplm_outcome_data<T>& data,
      const mat_type& w
    ) const;

    double partial_logprior(
      const mat_type& w
    ) const;

    /* Compute trace of [a' diag(v) b] */
    double mprod_trace(
      const mat_type& a,
      const mat_type& b,
      const vector_type& v
    ) const;

    mat_type grad_log_prior( const mat_type& w ) const;
    mat_type grad_log_likelihood(
      const gourd::gplm_outcome_data<T>& data,
      const mat_type& w
    ) const;

    mat_type grad_w(
      const gourd::gplm_outcome_data<T>& data,
      const mat_type& w
    );
    
    /*
    mat_type grad_g_fdiff(
      const gourd::gplm_outcome_data<T>& data,
      const T h = 1e-5,
      const int jmax = 5
    );
    */

    double update_omega_hmc(
      const gourd::gplm_outcome_data<T>& data,
      const int integrator_steps,
      const bool update = true
    );

    double update_omega_gibbs(
      const gourd::gplm_outcome_data<T>& data
    );

    double potential_energy() const;
    
    void sample_momentum_and_energy();

    void update_sigma_xi(
      const gourd::gplm_outcome_data<T>& data
    );
    
    void update_tau();

    void set_initial_omega(
      const gourd::gplm_outcome_data<T>& data
    );

    void set_initial_sigma(
      const gourd::gplm_outcome_data<T>& data
    );

    void set_initial_values(
      const gourd::gplm_outcome_data<T>& data
    );
    
  };
  // class surface_gplcov_model
  /* ****************************************************************/
  
};
// namespace gourd



/* Marginal log likelihood */
/* ******************************************************************/


double marginal_loglik( const long n, const double* x, void* data ) {
#ifdef GOURD_SINGLE_PRECISION
  using data_t  = gourd::gplm_outcome_data<float>;
  using param_t =
    typename abseil::covariance_functor<float, 3>::param_type;
#else
  using data_t = gourd::gplm_outcome_data<double>;
  using param_t =
    typename abseil::covariance_functor<double, 3>::param_type;
#endif
  //
  data_t* data_ptr = static_cast<data_t*>( data );
  
};


/* ******************************************************************/





template< typename T >
double gourd::surface_gplcov_model<T>::partial_loglik(
  const gourd::gplm_outcome_data<T>& data,
  const typename gourd::surface_gplcov_model<T>::mat_type& w
) const {
  // Can further speed up this computation?
  //  -> Already quite fast
  return mprod_trace(w, data.y(), sigma_sq_inv_) +
    -0.5 * mprod_trace(w, w, sigma_sq_inv_);
};

/*
 * Can evaluate the full log likelihood if needed for output:
 * y' (I_n \otimes \Sigma^-1) y = tr(Y' \Sigma^-1 Y) = tr(Y Y' \Sigma^-1)
 *   = data.yssq().adjoint() * sigma_sq_inv_;
 */


template< typename T >
double gourd::surface_gplcov_model<T>::partial_logprior(
  const typename gourd::surface_gplcov_model<T>::mat_type& w
) const {
  return -0.5 * tau_sq_inv_ * c_inv_.trqf(w);
};


/* Compute tr(a' diag(v) b) */
template< typename T >
double gourd::surface_gplcov_model<T>::mprod_trace(
  const typename gourd::surface_gplcov_model<T>::mat_type& a,
  const typename gourd::surface_gplcov_model<T>::mat_type& b,
  const typename gourd::surface_gplcov_model<T>::vector_type& v
) const {
  assert( a.cols() == b.cols() &&
	  a.rows() == b.rows() &&
	  a.rows() == v.size() &&
	  "surface_gplcov_model:matrix trace: dimensions must agree" );
  double trace = 0;
  for ( int j = 0; j < a.cols(); j++ )
    trace += static_cast<double>(
      (a.col(j).adjoint() * v.asDiagonal() * b.col(j)).coeff(0) );
  return trace;
};



template< typename T > inline
typename gourd::surface_gplcov_model<T>::mat_type
gourd::surface_gplcov_model<T>::grad_log_prior(
  const typename gourd::surface_gplcov_model<T>::mat_type& w
) const {
  return -tau_sq_inv_ * c_inv_.rmul( w );
};


template< typename T > inline
typename gourd::surface_gplcov_model<T>::mat_type
gourd::surface_gplcov_model<T>::grad_log_likelihood(
  const gourd::gplm_outcome_data<T>& data,
  const typename gourd::surface_gplcov_model<T>::mat_type& w
) const {
  return sigma_sq_inv_.asDiagonal() * ( data.y() - w );
};



template< typename T > inline
typename gourd::surface_gplcov_model<T>::mat_type
gourd::surface_gplcov_model<T>::grad_w(
  const gourd::gplm_outcome_data<T>& data,
  const typename gourd::surface_gplcov_model<T>::mat_type& w
) {
  return grad_log_likelihood(data, w) + grad_log_prior(w);
};





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplcov_model<T>::update(
  const gourd::gplm_outcome_data<T>& data,
  const bool update_learning_rate
) {
  update_tau();
  update_sigma_xi( data );
  /*
  const double alpha = update_omega_hmc( data, leapfrog_steps_ );
  if ( update_learning_rate )
    lr_.adapt( alpha );
  */
  const double alpha = update_omega_gibbs( data );
  std::cout << "\t\t<< \u03b1 = " << alpha
	    << "; \u03c4 = " << std::sqrt(1 / tau_sq_inv_)
	    << " >>"
	    << std::endl;
  return alpha;
};

// \alpha <==> \u03b1
// \tau   <==> \u03c4


template< typename T > 
double gourd::surface_gplcov_model<T>::update_omega_hmc(
  const gourd::gplm_outcome_data<T>& data,
  const int integrator_steps,
  const bool update
) {
  std::uniform_real_distribution<double> unif(0, 1);
  const T eps =
    lr_.eps( (update ? 0.9 + unif(gourd::urng()) * 0.2 : 1) );
  double log_prior_star, alpha;
  T k = 0.5;
  sample_momentum_and_energy();
  energy_initial_ = -partial_loglik(data, omega_) -
    partial_logprior(omega_);
  omega_star_ = omega_;
  momentum_ += k * eps * grad_w( data, omega_ );
  for ( int step = 0; step < integrator_steps; step++ ) {
    k = (step == (integrator_steps - 1)) ? 0.5 : 1;
    omega_star_.noalias() += eps * mass_.irmul( momentum_ );
    momentum_.noalias() += k * eps * grad_w( data, omega_star_ );
  }
  // ( momentum_ *= -1 )
  log_prior_star = partial_logprior(omega_star_);
  energy_proposal_ = -partial_loglik(data, omega_star_) -
    log_prior_star;
  alpha = std::exp( -energy_proposal_ - potential_energy() +
		    energy_initial_ + energy_momentum_ );
  alpha = isnan(alpha) ? 0 : alpha;
  if ( update  &&  unif(gourd::urng()) < alpha ) {
    omega_ = omega_star_;
    log_prior_kernel_omega_ = log_prior_star;
  }
  return (alpha > 1) ? 1 : alpha;
};
  // \epsilon <---> \u03b5




template< typename T > 
double gourd::surface_gplcov_model<T>::update_omega_gibbs(
  const gourd::gplm_outcome_data<T>& data
) {
  using cov_params = typename abseil::covariance_functor<T, 3>::param_type;
  // cov_ptr_->variance( 1 / tau_sq_inv_ );
  cov_params theta = cov_ptr_->param();
  theta[0] = 1 / tau_sq_inv_;
  cov_ptr_->param( theta );  // <- BAD
  //
  gourd::nnp_hess<T> var_(  // <- WRONG
    data.coordinates(),
    cov_ptr_,
    sigma_sq_inv_.cwiseInverse(),
    10.0,
    gourd::dist_code::great_circle
  );
  omega_ = var_.ihprod( sigma_sq_inv_.asDiagonal() * data.y() );
  log_prior_kernel_omega_ = partial_logprior(omega_);
  return 1.0;
};




template< typename T > 
void gourd::surface_gplcov_model<T>::update_sigma_xi(
  const gourd::gplm_outcome_data<T>& data
) {
  const int nloc = sigma_sq_inv_.size();
  const T shape = 0.5 * data.n() + 0.5;
  // const vector_type dsq = data.xsvd_d().cwiseAbs2();
  T rss;  // Residual sum of squares
  T sum_isig = 0;
  for ( int i = 0; i < nloc; i++ ) {
    rss = (data.y().row(i) - omega_.row(i)).cwiseAbs2().sum();
    //
    std::gamma_distribution<T> gam( shape, 1 / (xi_ + rss / 2) );
    sigma_sq_inv_.coeffRef(i) = gam(gourd::urng());
    sum_isig += sigma_sq_inv_.coeffRef(i);
  }
  /* Update prior on the rate of the sigma's */
  std::gamma_distribution<T> gam( 0.5 * nloc + 0.5,
				  1 / (1 + sum_isig) );
  xi_ = gam(gourd::urng());
};



template< typename T > 
void gourd::surface_gplcov_model<T>::update_tau() {
  const int s = omega_.rows();
  const int p = omega_.cols();
  const T shape = 0.5 * s * p + 1;
  const T rate = 0.5 * -log_prior_kernel_omega_ / tau_sq_inv_ + 1;
  std::gamma_distribution<T> gam( shape, 1 / rate );
  tau_sq_inv_ = gam(gourd::urng());
};
/* ****************************************************************/




template< typename T > 
void gourd::surface_gplcov_model<T>::sample_momentum_and_energy() {
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
double gourd::surface_gplcov_model<T>::potential_energy() const {
  return 0.5 * mass_.triqf(momentum_);
};






/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplcov_model<T>::set_initial_values(
  const gourd::gplm_outcome_data<T>& data
) {
  tau_sq_inv_ = 1;
  set_initial_sigma( data );
  set_initial_omega( data );
  update_tau(); /* Must be called AFTER set_initial_omega() */
  std::cout << "Initial sigma:\n"
	    << sigma_sq_inv_.head(10).cwiseSqrt().cwiseInverse().adjoint()
	    << "...\n" << std::endl;
};




template< typename T > 
bool gourd::surface_gplcov_model<T>::tune_initial_stepsize(
  const gourd::gplm_outcome_data<T>& data,
  const int maxit
) {
  bool tuning_needed = true;
  int it = 0;
  double alpha;
  while ( it < maxit && tuning_needed ) {
    alpha = update_omega_hmc( data, 1, false );
    tuning_needed = lr_.adjust_initial_value( alpha );
    it++;
  }
  if ( it >= maxit && tuning_needed ) {
    std::cerr << "surface_gplcov_model: initial HMC step size "
	      << "not found after " << it << " iterations\n"
	      << "\t(Final value was: " << lr_ << ")\n";
  }
  else {
    std::cerr << "surface_gplcov_model: HMC step size "
	      << "tuning took " << it << " iterations\n"
	      << "\tValue: " << lr_
	      << std::endl;
  }
  return !tuning_needed;
};




template< typename T > 
void gourd::surface_gplcov_model<T>::warmup(
  const gourd::gplm_outcome_data<T>& data,
  const int niter
) {
  assert(niter >= 0 &&
	 "surface_gplcov_model: negative warmup iterations");
  /*
  const bool eps_tuned = tune_initial_stepsize(data, niter/2);
  if ( !eps_tuned && niter/2 > 0 ) {
    std::cerr << "\t*** HMC step size tuning failed\n";
  }
  */
  for ( int i = 0; i < niter; i++ )
    update(data, true);
  lr_.fix();
};





template< typename T > 
void gourd::surface_gplcov_model<T>::profile(
  const gourd::gplm_outcome_data<T>& data,
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
    update_omega_hmc( data, 10 );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\omega(\\cdot)$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";


  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    grad_w( data, omega_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }  
  std::cout << "Evaluate gradient  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    partial_loglik( data, omega_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Evaluate log-likelihood  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    partial_logprior( omega_ );
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
  

  std::cout << "\\hline\n";
  std::cout << "\\end{tabular}\n"
	    << "\\end{table}\n";
  
  std::cout << std::endl;
};





template< typename T > 
void gourd::surface_gplcov_model<T>::set_initial_omega(
  const gourd::gplm_outcome_data<T>& data
) {
  omega_ = data.y();
  /* Jitter initial values */
  T tau = std::sqrt(
    (omega_.cwiseAbs2().colwise().mean() -
     omega_.colwise().mean().cwiseAbs2()).maxCoeff()
   );
  std::normal_distribution<T> normal(0, 0.1 * tau);
  for ( int i = 0; i < omega_.rows(); i++ ) {
    for ( int j = 0; j < omega_.cols(); j++ ) {
      omega_.coeffRef(i, j) += normal(gourd::urng());
    }
  }
  /* Set related initial values */
  omega_star_.resize( omega_.rows(), omega_.cols() );
  momentum_.resize( omega_.rows(), omega_.cols() );
  log_prior_kernel_omega_ = partial_logprior(omega_);
};




template< typename T > 
void gourd::surface_gplcov_model<T>::set_initial_sigma(
  const gourd::gplm_outcome_data<T>& data
) {
  sigma_sq_inv_.resize( data.nloc() );
  for ( int i = 0; i < data.nloc(); i++ ) {
    T rss = data.yssq().coeffRef(i) -
      std::pow(data.y().row(i).mean(), (T)2);
    sigma_sq_inv_.coeffRef(i) = data.n() / rss;
  }
  //
  xi_ = 1;
};
/* ****************************************************************/




/* ****************************************************************/
/* 
 *                               Getters
 */
template< typename T > inline
const typename gourd::surface_gplcov_model<T>::mat_type&
gourd::surface_gplcov_model<T>::omega() const {
  return omega_;
};


template< typename T > inline
typename gourd::surface_gplcov_model<T>::vector_type
gourd::surface_gplcov_model<T>::sigma() const {
  return sigma_sq_inv_.cwiseSqrt().cwiseInverse();
};


template< typename T > inline
T gourd::surface_gplcov_model<T>::tau() const {
  return 1 / std::sqrt( tau_sq_inv_ );
};


template< typename T > inline
T gourd::surface_gplcov_model<T>::xi() const {
  return xi_;
};
/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplcov_model<T>::surface_gplcov_model(
  const gourd::gplm_outcome_data<T>& data,
  abseil::covariance_functor<T, D>* const cov,
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
	 "surface_gplcov_model: negative HMC integrator steps");
  assert(nngp_radius >= 0 &&
	 "surface_gplcov_model: non-positive NNGP neighborhood radius");
  assert(mass_rad >= 0 &&
	 "surface_gplcov_model: non-positive mass matrix neighborhood");
  //
  /* Set Hessian */
  //
  cov_ptr_ = cov;
  //
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


#endif  // _GOURD_SURFACE_GPLCOV_MODEL_
