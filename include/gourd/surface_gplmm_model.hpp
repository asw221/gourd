
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

// #include <omp.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/math.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/covariance.hpp"
#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/rng.hpp"
#include "gourd/data/gplm_full_data.hpp"

// Profiling
#include "abseil/timer.hpp"
//


/*
 * This version uses a series of algorithms to estimate the posterior
 * of beta:
 *   - Get initial MAP estimate of beta, sigma
 *       * Conditional maximization
 *
 *   - (Get initial MAP estimate of omega)
 *
 *   - Subtract MAP estimate of omega from Y and sample from the 
 *     conditional posterior of beta, sigma
 *       * Using HMC and NNGP approximation
 *
 * TO DO
 * ----------
 * - Check for unused variables/functions
 * - Edit partial_loglik(...) and log_likelihood(...) methods to 
 *   evaluate using sufficient statistics
 */




#ifndef _GOURD_SURFACE_GPLMM_MODEL_
#define _GOURD_SURFACE_GPLMM_MODEL_




namespace gourd {

  /* ****************************************************************/
  /*! Gaussian process linear model for cortical surface data 
   *
   * 
   */
  template< typename T >
  class surface_gplmm_model {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;
    typedef typename Eigen::SparseMatrix<T> spmat_type;
    using cov_type = abseil::covariance_functor<T, 3>;

    /*
     * Needs to construct:
     *  - nnp_hess( vector<cartesian_coordinate>,
     *              covariance_functor, T radius, dist_code )
     *  - learning_rate( double eps0, double target_mh )
     */
    /*! (constructor) */
    template< size_t D >
    surface_gplmm_model(
      const gourd::gplm_full_data<T>& data,
      const abseil::covariance_functor<T, D>* const cov,
      const T nngp_radius,
      const gourd::dist_code distance,
      const int integrator_steps = 10,  /*!< HMC integrator steps */
      const double eps0 = 0.1,     /*!< HMC initial step size */
      const double mass_rad = 3,   /*!< HMC mass matrix n'hood radius */
      const double alpha0 = 0.65,  /*!< HMC target Metropolis-Hastings rate */
      const double eps_min = 1e-5, /*!< HMC minimum step size */
      const double g0 = 0.05,      /*!< HMC dual-averaging tuning parameter */
      const double t0 = 10,        /*!< HMC dual-averaging tuning parameter */
      const double k0 = 0.75       /*!< HMC dual-averaging tuning parameter */
    );

    surface_gplmm_model() = default;

    mat_type beta() const;
    vector_type sigma() const;
    T tau() const;
    T xi() const;
    
    const mat_type& omega() const;

    double learning_rate() const;
    double log_likelihood( const gourd::gplm_full_data<T>& data ) const;
    
    double update(
      const gourd::gplm_full_data<T>& data,
      const int monitor = 0,
      const bool update_learning_rate = false
    );

    double update_beta( const gourd::gplm_full_data<T>& data );
    
    void update_sigma_xi( const gourd::gplm_full_data<T>& data );
    
    void update_tau();

    bool tune_initial_stepsize(
      const gourd::gplm_full_data<T>& data,
      const int maxit = 100
    );

    void warmup(
      const gourd::gplm_full_data<T>& data,
      const int niter
    );

    void compute_map_estimate(
      const gourd::gplm_full_data<T>& data,
      const int maxit = 100,
      const T tol = 1e-6
    );


    void profile(
      const gourd::gplm_full_data<T>& data,
      const int nrep = 10
    );

    void beta( const mat_type& b );
    void sigma( const vector_type& s );
    
    
  protected:

    /* Model parameters */
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type gamma_star_;
    mat_type omega_;
    vector_type sigma_sq_inv_;  //
    T tau_sq_inv_;
    T xi_;
    gourd::nnp_hess<T> c_inv_;

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
      const gourd::gplm_full_data<T>& data,
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
      const gourd::gplm_full_data<T>& data,
      const mat_type& g
    ) const;

    mat_type grad_g(
      const gourd::gplm_full_data<T>& data,
      const mat_type& g
    );
    

    double update_gamma_hmc(
      const gourd::gplm_full_data<T>& data,
      const int integrator_steps,
      const bool update = true
    );

    double potential_energy() const;
    
    void sample_momentum_and_energy();

    void set_initial_gamma(
      const gourd::gplm_full_data<T>& data
    );

    void set_initial_sigma(
      const gourd::gplm_full_data<T>& data
    );

    void set_initial_values(
      const gourd::gplm_full_data<T>& data
    );

    void update_gamma_cmax(
      const gourd::gplm_full_data<T>& data
    );
    
    void update_sigma_xi_cmax(
      const gourd::gplm_full_data<T>& data
    );
    
    void update_omega_cmax(
      const gourd::gplm_full_data<T>& data
    );
    
  };
  // class surface_gplmm_model
  /* ****************************************************************/
  
};
// namespace gourd





template< typename T >
double gourd::surface_gplmm_model<T>::log_likelihood(
  const gourd::gplm_full_data<T>& data
) const {
  /*
   * Can evaluate the full log likelihood if needed for output:
   * y' (I_n \otimes \Sigma^-1) y = tr(Y' \Sigma^-1 Y) 
   *   = tr(Y Y' \Sigma^-1)
   *   = data.yssq().adjoint() * sigma_sq_inv_;
   */
  const double ysy = static_cast<double>(
    (data.yssq().adjoint() * sigma_sq_inv_).coeff(0) );
  //
  double logdet_sigma = 0;
  for ( int i = 0; i < sigma_sq_inv_.size(); i++ )
    logdet_sigma += std::log(sigma_sq_inv_.coeffRef(i));
  //
  const double normc = 0.5 * data.n() * data.nloc() *
    std::log(0.5 * num::inv_pi_v<double>) +
    0.5 * data.n() * logdet_sigma;
  return -0.5 * ysy + partial_loglik(data, gamma_) + normc;
};


template< typename T >
double gourd::surface_gplmm_model<T>::partial_loglik(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmm_model<T>::mat_type& g
) const {
  // Can further speed up this computation?
  //  -> Already quite fast
  return mprod_trace(g, data.yu() * data.xsvd_d().asDiagonal(),
		     sigma_sq_inv_) +
    -0.5 * mprod_trace(g, g * data.xsvd_d().cwiseAbs2().asDiagonal(),
		       sigma_sq_inv_);
};



template< typename T >
double gourd::surface_gplmm_model<T>::partial_logprior(
  const typename gourd::surface_gplmm_model<T>::mat_type& g
) const {
  return -0.5 * tau_sq_inv_ * c_inv_.trqf(g);
};


/* Compute tr(a' diag(v) b) */
template< typename T >
double gourd::surface_gplmm_model<T>::mprod_trace(
  const typename gourd::surface_gplmm_model<T>::mat_type& a,
  const typename gourd::surface_gplmm_model<T>::mat_type& b,
  const typename gourd::surface_gplmm_model<T>::vector_type& v
) const {
  assert( a.cols() == b.cols() &&
	  a.rows() == b.rows() &&
	  a.rows() == v.size() &&
	  "surface_gplmm_model:matrix trace: dimensions must agree" );
  double trace = 0;
  for ( int j = 0; j < a.cols(); j++ )
    trace += static_cast<double>(
      (a.col(j).adjoint() * v.asDiagonal() * b.col(j)).coeff(0) );
  return trace;
};



template< typename T > inline
typename gourd::surface_gplmm_model<T>::mat_type
gourd::surface_gplmm_model<T>::grad_log_prior(
  const typename gourd::surface_gplmm_model<T>::mat_type& g
) const {
  return -tau_sq_inv_ * c_inv_.rmul( g );
};


template< typename T > inline
typename gourd::surface_gplmm_model<T>::mat_type
gourd::surface_gplmm_model<T>::grad_log_likelihood(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmm_model<T>::mat_type& g
) const {
  return sigma_sq_inv_.asDiagonal() * (
    ( data.yu() - g * data.xsvd_d().asDiagonal() ) *
    data.xsvd_d().asDiagonal() );
};



template< typename T > inline
typename gourd::surface_gplmm_model<T>::mat_type
gourd::surface_gplmm_model<T>::grad_g(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmm_model<T>::mat_type& g
) {
  return grad_log_likelihood(data, g) + grad_log_prior(g);
};





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplmm_model<T>::update(
  const gourd::gplm_full_data<T>& data,
  const int monitor,
  const bool update_learning_rate
) {
  update_tau();
  update_sigma_xi( data );
  const double alpha = update_gamma_hmc( data, leapfrog_steps_ );
  if ( monitor > 0 ) {
    std::cout << "[" << monitor << "]\t\u03b1 = "
	      << std::setprecision(3) << std::fixed << alpha
	      << "\tloglik = " << log_likelihood(data)
	      << "\t\u03b5 = " << lr_
	      << std::endl;
  }
  if ( update_learning_rate ) { lr_.adapt( alpha ); }
  return alpha;
};

// \alpha <==> \u03b1
// \epsilon <==> \u03b5
// \tau   <==> \u03c4



template< typename T > 
double gourd::surface_gplmm_model<T>::update_beta(
  const gourd::gplm_full_data<T>& data
) {
  return update_gamma_hmc( data, leapfrog_steps_ );
};


template< typename T > 
double gourd::surface_gplmm_model<T>::update_gamma_hmc(
  const gourd::gplm_full_data<T>& data,
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
    gamma_star_.noalias() +=
      (eps / tau_sq_inv_) * mass_.irmul( momentum_ );
    momentum_.noalias() += k * eps * grad_g( data, gamma_star_ );
  }
  // ( momentum_ *= -1 )
  log_prior_star = partial_logprior(gamma_star_);
  energy_proposal_ = -partial_loglik(data, gamma_star_) -
    log_prior_star;
  alpha = std::exp( -energy_proposal_ - potential_energy() +
		    energy_initial_ + energy_momentum_ );
  // std::cout << "Proposal:  " << energy_proposal_ << "\n"
  // 	    << "Potential: " << potential_energy() << "\n"
  // 	    << "Initial:   " << energy_initial_ << "\n"
  // 	    << "Momentum:  " << energy_momentum_ << "\n"
  // 	    << std::endl;
  alpha = isnan(alpha) ? 0 : alpha;
  if ( update  &&  unif(gourd::urng()) < alpha ) {
    gamma_ = gamma_star_;
    log_prior_kernel_gamma_ = log_prior_star;
  }
  return (alpha > 1) ? 1 : alpha;
};




template< typename T > 
void gourd::surface_gplmm_model<T>::update_sigma_xi(
  const gourd::gplm_full_data<T>& data
) {
  const int nloc = sigma_sq_inv_.size();
  const T shape = 0.5 * data.n() + 0.5;
  // const vector_type dsq = data.xsvd_d().cwiseAbs2();
  T rss;  // Residual sum of squares
  T sum_isig = 0;
  for ( int i = 0; i < nloc; i++ ) {
    rss = data.yssq().coeffRef(i) +
      ( gamma_.row(i) * data.xsvd_d().asDiagonal() *
	(-2 * data.yu().row(i).adjoint() +
	 data.xsvd_d().asDiagonal() * gamma_.row(i).adjoint()
	 ) ).coeff(0);
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
void gourd::surface_gplmm_model<T>::update_tau() {
  const int s = gamma_.rows();
  const int p = gamma_.cols();
  const T shape = 0.5 * s * p + 1;
  const T rate = -log_prior_kernel_gamma_ / tau_sq_inv_ + 1;
  std::gamma_distribution<T> gam( shape, 1 / rate );
  const T draw = gam(gourd::urng());
  //
  log_prior_kernel_gamma_ *= draw / tau_sq_inv_;
  tau_sq_inv_ = draw;
};




/* --- Conditional maximization updates --- */


template< typename T > 
void gourd::surface_gplmm_model<T>::update_gamma_cmax(
  const gourd::gplm_full_data<T>& data
) {
  const mat_type z = sigma_sq_inv_.asDiagonal() * data.yu() *
    data.xsvd_d().asDiagonal();
  /* Decompositions must be computed block-wise */
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    T dsq = data.xsvd_d().coeff(j) * data.xsvd_d().coeff(j);
    spmat_type gram_gamma = tau_sq_inv_ * c_inv_.hessian();
    for ( int s = 0; s < gram_gamma.rows(); s++ ) {
      gram_gamma.coeffRef(s, s) += dsq * sigma_sq_inv_.coeffRef(s);
    }
    Eigen::SimplicialLDLT<spmat_type> ldl;
    ldl.analyzePattern( gram_gamma );
    ldl.factorize( gram_gamma );
    gamma_.col(j) = ldl.solve( z.col(j) );
  }
};


template< typename T > 
void gourd::surface_gplmm_model<T>::update_sigma_xi_cmax(
  const gourd::gplm_full_data<T>& data
) {
  const int nloc = sigma_sq_inv_.size();
  const T shape = 0.5 * data.n() * nloc + 0.5;
  const T rss =
    ( data.y() -
      gamma_ * data.xsvd_d().asDiagonal() * data.xsvd_u().adjoint() -
      omega_ ).rowwise().squaredNorm().sum();
  const T mode = (shape > T(1) ? (shape - 1) : shape) / (xi_ + rss / 2);
  T sum_isig = 0;
  for ( int s = 0; s < sigma_sq_inv_.size(); s++ ) {
    sigma_sq_inv_.coeffRef(s) = mode;
    sum_isig += sigma_sq_inv_.coeffRef(s);
  }
  /* Update prior on the rate of the sigma's */
  xi_ = (nloc > 1 ? (0.5 * nloc - 0.5) : (0.5 * nloc + 0.5)) /
    (1 + sum_isig);
  //
  // const T shape = 0.5 * data.n() + 0.5;
  // const vector_type rss =
  //   ( data.y() -
  //     gamma_ * data.xsvd_d().asDiagonal() * data.xsvd_u().adjoint() -
  //     omega_ ).rowwise().squaredNorm();
  //   sigma_sq_inv_.coeffRef(s) =
  //     (shape > T(1) ? (shape - 1) : shape) /
  //     (xi_ + rss.coeffRef(s) / 2);
};


template< typename T > 
void gourd::surface_gplmm_model<T>::update_omega_cmax(
  const gourd::gplm_full_data<T>& data
) {
  /* Compute decomposition */
  spmat_type gram_omega = tau_sq_inv_ * c_inv_.hessian();
  for ( int s = 0; s < gram_omega.rows(); s++ ) {
    gram_omega.coeffRef(s, s) += sigma_sq_inv_.coeffRef(s);
  }
  Eigen::SimplicialLDLT<spmat_type> ldl;
  ldl.analyzePattern( gram_omega );
  ldl.factorize( gram_omega );
  //
  /* Update individual effects */
  for ( int i = 0; i < data.n(); i++ ) {
    vector_type r = data.y(i) - gamma_ *
      (data.xsvd_d().asDiagonal() * data.xsvd_u().row(i).adjoint());
    omega_.col(i) = ldl.solve( sigma_sq_inv_.asDiagonal() * r );
  }
};
/* ****************************************************************/




template< typename T > 
void gourd::surface_gplmm_model<T>::sample_momentum_and_energy() {
  energy_momentum_ = 0;
  /* Profile outer loop as parallel */
  // #pragma omp parallel for reduction(+:energy_momentum_) shared(momentum_)
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
  momentum_ = std::sqrt(tau_sq_inv_) * mass_.hprod(momentum_).eval();
  /* ^^ Inefficient? Not really. Appears to be very marginally faster 
   * than preallocating the memory and copying into momentum_
   */
};



template< typename T > 
double gourd::surface_gplmm_model<T>::potential_energy() const {
  return (0.5 / tau_sq_inv_) * mass_.triqf(momentum_);
};






/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplmm_model<T>::set_initial_values(
  const gourd::gplm_full_data<T>& data
) {
  tau_sq_inv_ = 1;
  omega_ = mat_type::Zero( data.nloc(), data.n() );
  set_initial_sigma( data );
  set_initial_gamma( data );
  log_prior_kernel_gamma_ *= tau_sq_inv_;
  //
#ifndef NDEBUG
  std::cout << "Initial gamma:\n"
  	    << gamma_.topRows(10)
  	    << "\n" << std::endl;
  std::cout << "Initial sigma:\n"
  	    << sigma_sq_inv_.head(10).cwiseSqrt().cwiseInverse().adjoint()
  	    << "...\n" << std::endl;
#endif
};




template< typename T > 
bool gourd::surface_gplmm_model<T>::tune_initial_stepsize(
  const gourd::gplm_full_data<T>& data,
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
    std::cerr << "surface_gplmm_model: initial HMC step size "
	      << "not found after " << it << " iterations\n"
	      << "\t(Final value was: " << lr_ << ")\n";
  }
  else {
    std::cerr << "surface_gplmm_model: HMC step size "
	      << "tuning took " << it << " iterations\n"
	      << "\tValue: " << lr_
	      << std::endl;
  }
  return !tuning_needed;
};




template< typename T > 
void gourd::surface_gplmm_model<T>::warmup(
  const gourd::gplm_full_data<T>& data,
  const int niter
) {
  assert(niter >= 0 &&
	 "surface_gplmm_model: negative warmup iterations");
  log_prior_kernel_gamma_ = partial_logprior(gamma_);
  const bool eps_tuned = tune_initial_stepsize(data, niter/2);
  if ( !eps_tuned && niter/2 > 0 ) {
    std::cerr << "\t*** HMC step size tuning failed\n";
  }
  for ( int i = 0; i < niter; i++ )
    update(data, i+1, true);
  lr_.fix();
};



template< typename T > 
void gourd::surface_gplmm_model<T>::compute_map_estimate(
  const gourd::gplm_full_data<T>& data,
  const int maxit,
  const T tol
) {
  assert( maxit > 0  && "compute_map_estimate: maxit <= 0" );
  assert( tol > T(0) && "compute_map_estimate: tol <= 0" );
  std::cout << "Finding MAP estimates:\n";
  //
  T delta = static_cast<T>( HUGE_VAL );
  T xscale = 1;
  int i = 0;
  /* First maximize wrt gamma, sigma, xi */
  std::cout << "Maximizing wrt \u03b2(s), \u03c3(s):" << std::endl;
  while ( i < maxit && tol < (delta / xscale) ) {
    const mat_type g0 = gamma_;
    const vector_type s0 = sigma_sq_inv_;
    update_gamma_cmax( data );
    update_sigma_xi_cmax( data );
    delta = (gamma_ - g0).colwise().squaredNorm().sum() +
      (sigma_sq_inv_ - s0).squaredNorm();
    xscale = gamma_.colwise().template lpNorm<1>().sum() +
      sigma_sq_inv_.template lpNorm<1>();
    ++i;
    std::cout << "[" << i << "]  \u0394 = " << delta << std::endl;
  }
  if ( i == maxit && delta > tol ) {
    std::cerr << "\t***Warning: coefficients did not converge after "
	      << maxit << " iterations" << std::endl;
  }
  //
  /* Then maximize wrt omega, sigma, xi */
  delta = static_cast<T>( HUGE_VAL );
  xscale = 1;
  i = 0;
  std::cout << "\nMaximizing wrt \u03c9(s), \u03c3(s):" << std::endl;
  while ( i < maxit && tol < (delta / xscale) ) {
    const vector_type s0 = sigma_sq_inv_;
    update_omega_cmax( data );
    update_sigma_xi_cmax( data );
    delta = (sigma_sq_inv_ - s0).squaredNorm();
    xscale = sigma_sq_inv_.template lpNorm<1>();
    ++i;
    std::cout << "[" << i << "]  \u0394 = " << delta << std::endl;
  }
  if ( i == maxit && delta > tol ) {
    std::cerr << "\t***Warning: omega did not converge after "
	      << maxit << " iterations" << std::endl;
  }
  //
    // std::cout << "[" << i << "]:\n"
    // 	      << "beta =\n" << (gamma_.topRows(6) * vt_) << "...\n\n"
    // 	      << "sigma = " << sigma_sq_inv_.head(6).cwiseInverse().cwiseSqrt()
    // 	      << "\n" << std::endl;
    // std::cout << "[" << i << "]:\n"
    // 	      << "omega =\n" << omega_.topLeftCorner(6, 4) << "...\n\n"
    // 	      << "sigma = " << sigma_sq_inv_.head(6).cwiseInverse().cwiseSqrt()
    // 	      << "\n" << std::endl;
};



template< typename T > 
void gourd::surface_gplmm_model<T>::set_initial_gamma(
  const gourd::gplm_full_data<T>& data
) {
  vector_type di = data.xsvd_d().cwiseInverse();
  for ( int i = 0; i < di.size(); i++ ) {
    if ( isinf(di.coeffRef(i)) )  di.coeffRef(i) = 0;
  }
  gamma_ = data.yu() * di.asDiagonal();
  vt_ = data.xsvd_v().adjoint();
  /* Set related initial values */
  gamma_star_.resize( gamma_.rows(), gamma_.cols() );
  momentum_.resize( gamma_.rows(), gamma_.cols() );
};




template< typename T > 
void gourd::surface_gplmm_model<T>::set_initial_sigma(
  const gourd::gplm_full_data<T>& data
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
  //
  xi_ = 1;
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
typename gourd::surface_gplmm_model<T>::mat_type
gourd::surface_gplmm_model<T>::beta() const {
  return gamma_ * vt_;
};


template< typename T > inline
typename gourd::surface_gplmm_model<T>::vector_type
gourd::surface_gplmm_model<T>::sigma() const {
  return sigma_sq_inv_.cwiseSqrt().cwiseInverse();
};


template< typename T > inline
T gourd::surface_gplmm_model<T>::tau() const {
  return 1 / std::sqrt( tau_sq_inv_ );
};


template< typename T > inline
T gourd::surface_gplmm_model<T>::xi() const {
  return xi_;
};

template< typename T >
double gourd::surface_gplmm_model<T>::learning_rate() const {
  return lr_;
};


template< typename T >
const typename gourd::surface_gplmm_model<T>::mat_type&
gourd::surface_gplmm_model<T>::omega() const {
  return omega_;
};
/* ****************************************************************/




/* ****************************************************************/
/* 
 *                               Setters
 */
template< typename T > 
void gourd::surface_gplmm_model<T>::beta(
  const typename gourd::surface_gplmm_model<T>::mat_type& b
) {
  if ( b.rows() != gamma_.rows() || b.cols() != gamma_.cols() ) {
    throw std::domain_error( "Cannot change parameter dimensions" );
  }
  gamma_ = b * vt_.adjoint();
};


template< typename T > 
void gourd::surface_gplmm_model<T>::sigma(
  const typename gourd::surface_gplmm_model<T>::vector_type& s
) {
  if ( s.size() != sigma_sq_inv_.size() ) {
    throw std::domain_error( "Cannot change parameter dimensions" );
  }
  sigma_sq_inv_ = s.cwiseAbs2().cwiseInverse();
};
/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplmm_model<T>::surface_gplmm_model(
  const gourd::gplm_full_data<T>& data,
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
	 "surface_gplmm_model: negative HMC integrator steps");
  assert(nngp_radius >= 0 &&
	 "surface_gplmm_model: non-positive NNGP neighborhood radius");
  assert(mass_rad >= 0 &&
	 "surface_gplmm_model: non-positive mass matrix neighborhood");
  //
  // Make copy of cov
  const typename cov_type::param_type theta0 = cov->param();
  std::unique_ptr<cov_type> cov_copy;
  gourd::init_cov(
    cov_copy, gourd::get_cov_code(cov), theta0.cbegin(), theta0.cend()
  );
  tau_sq_inv_ = 1 / cov->variance();
  cov_copy->variance(1);
  /* Set Hessian */
  c_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov_copy.get(),
    nngp_radius,
    distance
  );
  /* Set Mass matrix */
  mass_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov_copy.get(),
    mass_rad,
    distance
  );
  /* Set HMC learning rate/step size */
  lr_ = abseil::learning_rate( eps0, alpha0, eps_min, g0, t0, k0 );
  leapfrog_steps_ = integrator_steps;
  /* Initialize parameters */
  set_initial_values( data );
};






template< typename T > 
void gourd::surface_gplmm_model<T>::profile(
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





#endif  // _GOURD_SURFACE_GPLMM_MODEL_
