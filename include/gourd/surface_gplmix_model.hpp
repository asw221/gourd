
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/math.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/covariance.hpp"
#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/rng.hpp"
#include "gourd/surface_gpl_covariance.hpp"

#include "gourd/data/gplm_full_data.hpp"

// Profiling
#include "abseil/timer.hpp"
//


/*
 * This version uses a series of algorithms to estimate the posterior
 * of beta:
 *   - Get initial MAP estimate of beta
 *       * Conditional maximization
 *
 *   - Condition on beta and optimize over omega's covariance
 *     parameters 
 *       * routines from "gourd/surface_gpl_covariance.hpp"
 *
 *   - Condition on MAP estimates of all of the residual covariance
 *     parameters to generate approximate samples of beta from its 
 *     marginal posterior
 *       * Using HMC and NNGP approximation
 *
 * TO DO
 * ----------
 * - Check for unused variables/functions
 * - Edit partial_loglik(...) and log_likelihood(...) methods to 
 *   evaluate using sufficient statistics
 */





#ifndef _GOURD_SURFACE_GPLMIX_MODEL_
#define _GOURD_SURFACE_GPLMIX_MODEL_




namespace gourd {

  /* ****************************************************************/
  /*! Gaussian process linear mixed model for cortical surface data 
   *
   * 
   */
  template< typename T >
  class surface_gplmix_model {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;
    typedef typename Eigen::SparseMatrix<T> spmat_type;
    using cov_type = abseil::covariance_functor<T, 3>;

    
    /*! (constructor) */
    template< size_t D >
    surface_gplmix_model(
      const gourd::gplm_full_data<T>& data,
      const abseil::covariance_functor<T, D>* const cov,
      const T nngp_radius,
      const gourd::dist_code distance,
      const double mixef_rad = 4,  /*!< Random intercept n'hood radius */
      const int integrator_steps = 10,  /*!< HMC integrator steps */
      const double eps0 = 0.1,     /*!< HMC initial step size */
      const double mass_rad = 3,   /*!< HMC mass matrix n'hood radius */
      const double alpha0 = 0.65,  /*!< HMC target Metropolis-Hastings rate */
      const double eps_min = 1e-5, /*!< HMC minimum step size */
      const double g0 = 0.05,      /*!< HMC dual-averaging tuning parameter */
      const double t0 = 10,        /*!< HMC dual-averaging tuning parameter */
      const double k0 = 0.75       /*!< HMC dual-averaging tuning parameter */
    );
    /*
     * Needs to construct:
     *  - nnp_hess( vector<cartesian_coordinate>,
     *              covariance_functor, T radius, dist_code )
     *  - learning_rate( double eps0, double target_mh )
     */

    surface_gplmix_model() = default;

    T eta() const;
    T sigma() const;
    T tau() const;
    const mat_type& beta() const;

    mat_type smdiag( const gourd::gplm_full_data<T>& data ) const;
    /*<! Compute the diagonal of the smoothing/hat matrix. EXTREMELY slow */
    
    double log_likelihood(
      const gourd::gplm_full_data<T>& data
    ) const;
    
    double update(
      const gourd::gplm_full_data<T>& data,
      const int monitor = 0,
      const bool update_learning_rate = false
    );
    
    void compute_map_estimate(
      const gourd::gplm_full_data<T>& data,
      const int maxit = 100,
      const T tol = 1e-8
    );
    
    void update_error_terms(
      const gourd::gplm_full_data<T>& data
    );

    bool tune_initial_stepsize(
      const gourd::gplm_full_data<T>& data,
      const int maxit = 100
    );

    void warmup(
      gourd::gplm_full_data<T>& data,
      const int niter_optim,
      const int niter_burnin,
      const T xtol_optim
    );

    void profile(
      const gourd::gplm_full_data<T>& data,
      const int nrep = 10
    );

    //
    void beta( const mat_type& b );
    
    
  protected:

    /* Model parameters */
    mat_type beta_;
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type gamma_star_;
    T sigma_sq_inv_;
    T tau_sq_inv_;
    T eta_sq_inv_;
    //
    // spmat_type resid_gram_;
    gourd::nnp_hess<T> resid_gram_;
    // T xi_;
    gourd::nnp_hess<T> c_inv_;

    /* Parameters related to updates */
    mat_type momentum_;
    mat_type vt_;                /* X = U D ~Vt~ */
    double energy_initial_;      /* Partial log posterior */
    double energy_momentum_;
    double energy_proposal_;
    //
    int leapfrog_steps_;
    abseil::learning_rate lr_;  //
    gourd::nnp_hess<T> mass_;
    //
    T gammassq_;
    T residssq_;

    //
    std::unique_ptr<cov_type> re_cov_ptr_;
    gourd::cov_code re_cov_code_;
    gourd::dist_code re_dist_code_;
    double re_rad_;
    //


    /* Compute unnormalized log likelihood */
    double partial_loglik(
      const gourd::gplm_full_data<T>& data,
      const mat_type& g
    ) const;

    double partial_logprior(
      const mat_type& g
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

    

    void update_gamma_cmax(
      const gourd::gplm_full_data<T>& data
    );
    
    void update_sigma_cmax(
      const gourd::gplm_full_data<T>& data
    );
    
    void update_tau_cmax(
      const gourd::gplm_full_data<T>& data
    );

    void update_resid_gram(
      const gourd::gplm_full_data<T>& data
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
    
  };
  // class surface_gplmix_model
  /* ****************************************************************/
  
};
// namespace gourd



template< typename T >
double gourd::surface_gplmix_model<T>::log_likelihood(
  const gourd::gplm_full_data<T>& data
) const {
  return 0.5 * data.n() * 
    (data.nloc() * std::log(0.5 * num::inv_pi_v<double>) +
     resid_gram_.ldet()) +
    partial_loglik(data, gamma_);
};



template< typename T >
double gourd::surface_gplmix_model<T>::partial_loglik(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) const {
  const mat_type ud = data.xsvd_u() * data.xsvd_d().asDiagonal();
  T lk = 0;
  for ( int i = 0; i < data.n(); i++ ) {
    vector_type resid = data.y(i) - (g * ud.row(i).adjoint());
    // lk += ( resid.adjoint() * resid_gram_ * resid ).coeff(0);
    lk += resid_gram_.qf( resid );
  }
  return -0.5 * lk;
};


/*
 * Can evaluate the full log likelihood if needed for output:
 * y' (I_n \otimes \Sigma^-1) y = tr(Y' \Sigma^-1 Y) = tr(Y Y' \Sigma^-1)
 *   = data.yssq().adjoint() * sigma_sq_inv_;
 */


template< typename T >
double gourd::surface_gplmix_model<T>::partial_logprior(
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) const {
  return -0.5 * tau_sq_inv_ * eta_sq_inv_ * c_inv_.trqf(g);
};




template< typename T > inline
typename gourd::surface_gplmix_model<T>::mat_type
gourd::surface_gplmix_model<T>::grad_log_prior(
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) const {
  return -tau_sq_inv_ * eta_sq_inv_ * c_inv_.rmul( g );
};

/*
  g = (V' o I) b
  b ~ N(0, T o C)  ==>  g ~ N(0, V' T V o C)
    ==>  -0.5 g' (V' T^-1 V o C^-1) g
  grad_g = -(V' T^-1 V o C^-1) g <==> -C^-1 G V' T^-1 V
 */




template< typename T > inline
typename gourd::surface_gplmix_model<T>::mat_type
gourd::surface_gplmix_model<T>::grad_log_likelihood(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) const {
  return resid_gram_.rmul(
    ( data.yu() - g * data.xsvd_d().asDiagonal() ) *
    data.xsvd_d().asDiagonal()
  );
};




template< typename T > inline
typename gourd::surface_gplmix_model<T>::mat_type
gourd::surface_gplmix_model<T>::grad_g(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) {
  return grad_log_likelihood(data, g) + grad_log_prior(g);
};





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplmix_model<T>::update(
  const gourd::gplm_full_data<T>& data,
  const int monitor,
  const bool update_learning_rate
) {
  // update_error_terms( data );
  const double alpha = update_gamma_hmc( data, leapfrog_steps_ );
  if ( monitor > 0 ) {
    std::cout << "[" << monitor << "]\t\u03b1 = "
	      << std::setprecision(3) << std::fixed << alpha
	      << "\tloglik = " << partial_loglik(data, gamma_)
	      << "\t\u03b5 = " << lr_
	      << std::endl;
  }
  //
  // std::cout << "B =\n" << beta_.topRows(10) << "\n";
  // std::cout << "\u03c4\u00b2 = " << (1 / tau_sq_inv_) << "\n";
  // std::cout << "\u03b7\u00b2 = " << (1 / eta_sq_inv_) << "\n";
  // std::cout << "\u03c3\u00b2 = " << (1 / sigma_sq_inv_) << "\n";
  // std::cout << "\t\t<< \u03b5 = " << lr_ << " >>\n";
  // std::cout << "\t\t<< loglik = " << partial_loglik(data, gamma_) << " >>\n";
  // std::cout << "\t\t<< \u03b1 = " << alpha << " >>" << std::endl;
  //
  if ( update_learning_rate ) { lr_.adapt( alpha ); }
  return alpha;
};

// \alpha <==> \u03b1
// \tau   <==> \u03c4



template< typename T > 
double gourd::surface_gplmix_model<T>::update_gamma_hmc(
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
      (eps / (tau_sq_inv_ * eta_sq_inv_)) * mass_.irmul( momentum_ );
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
    beta_ = gamma_ * vt_;
  }
  return (alpha > 1) ? 1 : alpha;
};



template< typename T > 
void gourd::surface_gplmix_model<T>::update_gamma_cmax(
  const gourd::gplm_full_data<T>& data
) {
  const mat_type z = sigma_sq_inv_ * data.yu() *
    data.xsvd_d().asDiagonal();
  gammassq_ = 0;
  /* Decompositions must be computed block-wise */
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    T dsq = data.xsvd_d().coeff(j) * data.xsvd_d().coeff(j);
    spmat_type gram_gamma = (tau_sq_inv_ * eta_sq_inv_) *
      c_inv_.hessian();
    for ( int s = 0; s < gram_gamma.rows(); s++ ) {
      gram_gamma.coeffRef(s, s) += dsq * sigma_sq_inv_;
    }
    Eigen::SimplicialLDLT<spmat_type> ldl;
    ldl.analyzePattern( gram_gamma );
    ldl.factorize( gram_gamma );
    gamma_.col(j) = ldl.solve( z.col(j) );
    gammassq_ += c_inv_.qf( gamma_.col(j) );
  }
  beta_ = gamma_ * vt_;
  //
  residssq_ = 0;
  for ( int i = 0; i < data.n(); i++ ) {
    residssq_ += (data.y(i) - beta_ * data.x().row(i).adjoint())
      .squaredNorm();
  }
  // std::cout << "\tSUM(gamma^2) = " << gammassq_ << std::endl;
};



template< typename T >
void gourd::surface_gplmix_model<T>::update_error_terms(
  const gourd::gplm_full_data<T>& data
) {
  update_simga_cmax( data );
  update_tau_cmax( data );
  update_eta_cmax( data );
  //
  update_resid_gram( data );
};





template< typename T >
void gourd::surface_gplmix_model<T>::update_sigma_cmax(
  const gourd::gplm_full_data<T>& data
) {
  const T shape = 0.5 + 0.5 * data.n() * data.nloc();
  const T rate = 0.5 + 0.5 * residssq_;
  /* Set sigma^-2 to its conditional posterior mode if the mode
   * exists, and to its conditional posterior mean if otherwise */
  sigma_sq_inv_ = (shape >= 1) ? (shape - 1) : shape;
  sigma_sq_inv_ /= rate;
  // sigma_sq_inv_ = (shape + 1) / rate;
};



template< typename T >
void gourd::surface_gplmix_model<T>::update_tau_cmax(
  const gourd::gplm_full_data<T>& data
) {
  const T shape = 0.5 + 0.5 * data.nloc() * gamma_.cols();
  const T rate = 0.5 + 0.5 * gammassq_ * eta_sq_inv_;
  // std::cout << "\tTAU: shape = " << shape << ", rate = " << rate << std::endl;
  tau_sq_inv_ = (shape >= 1) ? (shape - 1) : shape;
  tau_sq_inv_ /= rate;
  // tau_sq_inv_ = (shape + 1) / rate;
};


template< typename T >
void gourd::surface_gplmix_model<T>::update_resid_gram(
  const gourd::gplm_full_data<T>& data
) {
  re_cov_ptr_->variance( 1 / eta_sq_inv_ );
  resid_gram_ = gourd::nnp_hess<T>(
    data.coordinates(),
    re_cov_ptr_.get(),
    vector_type::Constant(data.nloc(), 1/sigma_sq_inv_),
    re_rad_,
    re_dist_code_
  );
};




template< typename T >
void gourd::surface_gplmix_model<T>::compute_map_estimate(
  const gourd::gplm_full_data<T>& data,
  const int maxit,
  const T tol
) {
  assert( maxit > 0  && "compute_map_estimate: maxit <= 0" );
  assert( tol > T(0) && "compute_map_estimate: tol <= 0" );
  std::cout << "Finding MAP estimates:\n";
  update_resid_gram( data );
  /* First maximize wrt gamma, tau, sigma */
  T delta = static_cast<T>( HUGE_VAL );
  T llkprev = -log_likelihood(data);
  int i = 0;
  while ( i < maxit && delta > tol ) {
    T dllk = -llkprev;
    //
    update_gamma_cmax( data );
    update_sigma_cmax( data );
    // update_tau_cmax( data );
    //
    llkprev = log_likelihood( data );
    dllk += llkprev;
    delta = (dllk < T(0)) ? -dllk : dllk;
    i++;
    //
    std::cout << "[" << i << "]  "
	      << "loglik = " << llkprev << ";  "
	      << "\u03c4\u00b2 = " << (1 / tau_sq_inv_) << ";  "
	      << "\u03b7\u00b2 = " << (1 / eta_sq_inv_) << ";  "
	      << "\u03c3\u00b2 = " << (1 / sigma_sq_inv_)
	      << std::endl;
    // //
    // std::cout << "beta:\n" << beta_.topRows(10)
    // 	      << "...\n" << std::endl;
    // //
  }
  std::cout << "  ~~~\n"
  	    << "loglik = " << llkprev << ";  "
  	    << "\u03c4\u00b2 = " << (1 / tau_sq_inv_) << ";  "
  	    << "\u03b7\u00b2 = " << (1 / eta_sq_inv_) << ";  "
  	    << "\u03c3\u00b2 = " << (1 / sigma_sq_inv_)
  	    << std::endl;
};

/* **************************************************************** */




template< typename T > 
void gourd::surface_gplmix_model<T>::sample_momentum_and_energy() {
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
  momentum_ = std::sqrt(tau_sq_inv_ * eta_sq_inv_) *
    mass_.hprod(momentum_).eval();
  /* ^^ Inefficient? Not really. Appears to be very marginally faster 
   * than preallocating the memory and copying into momentum_
   */
};



template< typename T > 
double gourd::surface_gplmix_model<T>::potential_energy() const {
  return (0.5 / (tau_sq_inv_ * eta_sq_inv_)) * mass_.triqf(momentum_);
};






/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_values(
  const gourd::gplm_full_data<T>& data
) {
  // tau_sq_inv_ = 1;
  eta_sq_inv_ = 1;
  gammassq_ = 0; residssq_ = 0;
  set_initial_gamma( data );
  //
  /* Must be called AFTER set_initial_gamma() */
  // update_tau_cmax( data );
  set_initial_sigma( data );
  // update_tau(); /* Must be called AFTER set_initial_gamma() */
  //
#ifndef NDEBUG
  std::cout << "Initial gamma:\n" << gamma_.topRows(10)
  	    << "...\n" << std::endl;
#endif
};




template< typename T > 
bool gourd::surface_gplmix_model<T>::tune_initial_stepsize(
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
    std::cerr << "surface_gplmix_model: initial HMC step size "
	      << "not found after " << it << " iterations\n"
	      << "\t(Final value was: " << lr_ << ")\n";
  }
  else {
    std::cerr << "surface_gplmix_model: HMC step size "
	      << "tuning took " << it << " iterations\n"
	      << "\tValue: " << lr_
	      << std::endl;
  }
  return !tuning_needed;
};




template< typename T > 
void gourd::surface_gplmix_model<T>::warmup(
  gourd::gplm_full_data<T>& data,
  const int niter_optim,
  const int niter_burnin,
  const T xtol_optim
) {
  assert(niter_optim >= 0 &&
	 "surface_gplmix_model: negative optimization iterations");
  assert(niter_burnin >= 0 &&
	 "surface_gplmix_model: negative warmup iterations");
  /* Iteratively compute MAP estimate */
  compute_map_estimate( data, niter_optim, xtol_optim );
  //
  for ( int i = 0; i < data.n(); i++ ) {
    data.y_ref().col(i).noalias() -=
      beta_ * data.x().row(i).adjoint();
  }
  gourd::sgpl::data_pack pack;
  pack.cov_ptr = re_cov_ptr_.get();
  pack.data_ptr = &data;
  pack.neighborhood_radius = re_rad_;
  pack.distance_metric = re_dist_code_;
  pack.marginal_var = data.y().cwiseAbs2().rowwise().mean() -
    data.y().rowwise().mean().cwiseAbs2();
  gourd::sgpl::optim_output opt =
    gourd::sgpl::optimize_cov_params(pack, 3, 1e3, 1e-8, 3);
  if ( opt.code != gourd::sgpl::optim_code::success ) {
    std::cerr << "Optimizer reported error state ("
	      << opt.code << ")" << std::endl;
  }
  for ( int i = 0; i < data.n(); i++ ) {
    data.y_ref().col(i).noalias() +=
      beta_ * data.x().row(i).adjoint();
  }
  eta_sq_inv_ = 1 / re_cov_ptr_->variance();
  tau_sq_inv_ /= eta_sq_inv_;
  sigma_sq_inv_ = 1 /
    ((1 + 1e-5) * pack.marginal_var.mean() - re_cov_ptr_->variance());
  resid_gram_ = gourd::nnp_hess<T>(
    data.coordinates(),
    re_cov_ptr_.get(),
    pack.marginal_var,
    re_rad_,
    re_dist_code_
  );
  std::cout << re_cov_ptr_->param() << std::endl;
  //
  /* Run HMC burnin */
  if ( niter_burnin > 0 ) {
    const bool eps_tuned = tune_initial_stepsize(data, 20);
    if ( !eps_tuned ) {
      std::cerr << "\t*** HMC step size tuning failed\n";
    }
    for ( int i = 0; i < niter_burnin; i++ )
      update(data, i+1, true);
    lr_.fix();
  }
};







template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_gamma(
  const gourd::gplm_full_data<T>& data
) {
  // using spmat_t = typename Eigen::SparseMatrix<T>;
  vector_type di = data.xsvd_d().cwiseInverse();
  for ( int i = 0; i < di.size(); i++ ) {
    if ( isinf(di.coeffRef(i)) )  di.coeffRef(i) = 0;
  }
  gamma_ = data.yu() * di.asDiagonal();
  vt_ = data.xsvd_v().adjoint();
  /* Set related initial values */
  beta_ = gamma_ * vt_;
  gamma_star_.resize( gamma_.rows(), gamma_.cols() );
  momentum_.resize( gamma_.rows(), gamma_.cols() );
  //
  gammassq_ = 0;
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    gammassq_ += c_inv_.qf( gamma_.col(j) );
  }
};




template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_sigma(
  const gourd::gplm_full_data<T>& data
) {
  const int ns = data.n() * data.nloc();
  T first = 0, second = 0;  // moments
  for ( int i = 0; i < data.n(); i++ ) {
    vector_type ri = data.y(i) - beta_ * data.x().row(i).adjoint();
    first += ri.sum();
    second += ri.cwiseAbs2().sum();
  }
  first /= ns; second /= ns;
  sigma_sq_inv_ = 1 / (second - first * first);
  // int df = data.n() - data.p();
  // df = ( df < 1 ) ? 1 : df;
  // sigma_sq_inv_.resize( data.nloc() );
  // for ( int i = 0; i < data.nloc(); i++ ) {
  //   T eta = (data.xsvd_u() * data.yu().row(i).adjoint()).sum();
  //   sigma_sq_inv_.coeffRef(i) =
  //     df / ( data.yssq().coeffRef(i) - eta * eta / df );
  //   if ( isnan(sigma_sq_inv_.coeffRef(i)) ||
  // 	 sigma_sq_inv_.coeffRef(i) <= 0 ) {
  //     sigma_sq_inv_.coeffRef(i) = 1;
  //   }
  // }
  // //
  // xi_ = 1;
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
const typename gourd::surface_gplmix_model<T>::mat_type&
gourd::surface_gplmix_model<T>::beta() const {
  return beta_;
};


template< typename T > inline
T gourd::surface_gplmix_model<T>::eta() const {
  return std::sqrt( re_cov_ptr_->variance() );
};


template< typename T > inline
T gourd::surface_gplmix_model<T>::sigma() const {
  return std::sqrt( 1 / sigma_sq_inv_ );
};


template< typename T > inline
T gourd::surface_gplmix_model<T>::tau() const {
  return std::sqrt( 1 / tau_sq_inv_ );
};



template< typename T >
typename gourd::surface_gplmix_model<T>::mat_type
gourd::surface_gplmix_model<T>::smdiag(
  const gourd::gplm_full_data<T>& data					     
) const {
  // Compute hat matrix diagonal
  mat_type dhat = mat_type::Zero(data.nloc(), data.n());
  const spmat_type sigma_inv = resid_gram_.hessian();
  for ( int j = 0; j < data.xsvd_d().size(); j++ ) {
    double dj = data.xsvd_d().coeffRef(j);
    spmat_type hess = (tau_sq_inv_ * eta_sq_inv_) * c_inv_.hessian();
    hess += (dj * dj) * sigma_inv;
    Eigen::SimplicialLDLT<spmat_type> ldl( hess );
    for ( int k = 0; k < dhat.size(); k++ ) {
      int s = k % data.nloc();  // location index
      int i = k / data.nloc();  // subject index
      double uij = data.xsvd_u().coeffRef(i, j);
      vector_type v = (dj * uij) * sigma_inv.col( s );
      vector_type w = ldl.solve( v );
      dhat.coeffRef(s, i) += (dj * uij) * w.coeff( s );
    }
  }
  return dhat;
};

/* ****************************************************************/



/* ****************************************************************/
/* 
 *                               Setters
 */
template< typename T > 
void gourd::surface_gplmix_model<T>::beta(
  const typename gourd::surface_gplmix_model<T>::mat_type& b
) {
  if ( b.rows() != gamma_.rows() || b.cols() != gamma_.cols() ) {
    throw std::domain_error( "Cannot change parameter dimensions" );
  }
  beta_  = b;
  gamma_ = b * vt_.adjoint();
};
/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplmix_model<T>::surface_gplmix_model(
  const gourd::gplm_full_data<T>& data,
  const abseil::covariance_functor<T, D>* const cov,
  const T nngp_radius,
  const gourd::dist_code distance,
  const double mixef_rad,
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
	 "surface_gplmix_model: negative HMC integrator steps");
  assert(nngp_radius >= 0 &&
	 "surface_gplmix_model: non-positive NNGP neighborhood radius");
  assert(mixef_rad >= 0 &&
	 "surface_gplmix_model: non-positive NNGP neighborhood radius");
  assert(mass_rad >= 0 &&
	 "surface_gplmix_model: non-positive mass matrix neighborhood");
  //
  const typename cov_type::param_type re_theta = cov->param();
  gourd::init_cov(
    re_cov_ptr_,
    gourd::get_cov_code(cov),
    re_theta.cbegin(),
    re_theta.cend()
  );
  re_dist_code_ = distance;
  re_rad_ = mixef_rad;
  re_cov_ptr_->variance( 1 );
  tau_sq_inv_ = 1 / cov->variance();
  //
  /* Set Hessian */
  c_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    re_cov_ptr_.get(),
    nngp_radius,
    distance
  );
  //
  resid_gram_ = gourd::nnp_hess<T>(
    data.coordinates(),
    re_cov_ptr_.get(),
    mixef_rad,
    distance
  );
  //
  /* Set Mass matrix */
  mass_ = gourd::nnp_hess<T>(
    data.coordinates(),
    re_cov_ptr_.get(),
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
void gourd::surface_gplmix_model<T>::profile(
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
    update_gamma_hmc( data, leapfrog_steps_ );
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
    update_error_terms( data );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Update $\\tau^{-2}, \\eta^{-2}, \\sigma^{-2}$  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";
  

  std::cout << "\\hline\n";
  std::cout << "\\end{tabular}\n"
	    << "\\end{table}\n";
  
  std::cout << std::endl;
};


#endif  // _GOURD_SURFACE_GPLMIX_MODEL_
