
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>  // std::iota
#include <random>
#include <vector>

// #include <omp.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/gplm_data.hpp"
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
  /*! Gaussian process linear mixed effects model for cortical 
   * surface data
   *
   * Special case: fixed effects, plus random patient-level spatial
   * intercepts
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
      const gourd::gplm_data<T>& data,
      const abseil::covariance_functor<T, D>* const cov,
      const T nngp_radius,
      const gourd::dist_code distance,
      const double eps0 = 0.1,     /*!< HMC initial step size */
      const double alpha0 = 0.65,  /*!< HMC target Metropolis-Hastings rate */
      const double eps_min = 1e-5, /*!< HMC minimum step size */
      const double g0 = 0.05,      /*!< HMC dual-averaging tuning parameter */
      const double t0 = 10,        /*!< HMC dual-averaging tuning parameter */
      const double k0 = 0.75       /*!< HMC dual-averaging tuning parameter */
    );

    surface_gplme_model() = default;    
    
    mat_type beta() const;
    vector_type sigma() const;
    T tau() const;
    T xi() const;
    
    double update(
      const gourd::gplm_data<T>& data,
      const int integrator_steps,
      const bool update_learning_rate
    );

    bool tune_initial_stepsize(
      const gourd::gplm_data<T>& data,
      const int maxit = 100
    );

    void warmup(
      const gourd::gplm_data<T>& data,
      const int niter,
      const int itegrator_steps
    );

    void profile(
      const gourd::gplm_data<T>& data,
      const int nrep = 10
    );
    
    
  protected:

    /* Model parameters */
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type gamma_star_;
    mat_type ru_;               /* R U:  R = (Y - W); X = U D Vt */
    mat_type w_;                /* Random spatial intercepts */
    vector_type sigma_sq_inv_;  //
    vector_type zeta_inv_;      /* zeta_i = var(W_i) */
    T tau_sq_inv_;
    T xi_;
    gourd::nnp_hess<T> c_inv_;

    /* Parameters related to HMC */
    gourd::nnp_hess<T> mass_;
    mat_type momentum_;
    mat_type vt_;                /* X = U D ~Vt~ */
    double energy_initial_;      /* Partial log posterior */
    double energy_momentum_;
    double energy_proposal_;
    double log_prior_kernel_gamma_;  /* Prior on gamma_: log kernel */
    abseil::learning_rate lr_;  //

    /* Etc */
    std::vector<int> order_update_w_;


    double partial_loglik_g(
      const gourd::gplm_data<T>& data,
      const mat_type& g
    ) const;

    double partial_logprior_g(
      const mat_type& g
    ) const;

    /* Compute trace of [a' diag(v) b] */
    double mprod_trace(
      const mat_type& a,
      const mat_type& b,
      const vector_type& v
    ) const;

    mat_type grad_g_log_prior( const mat_type& g ) const;
    mat_type grad_g_log_likelihood(
      const gourd::gplm_data<T>& data,
      const mat_type& g
    ) const;

    mat_type grad_g(
      const gourd::gplm_data<T>& data,
      const mat_type& g
    );
    
    /*
    mat_type grad_g_fdiff(
      const gourd::gplm_data<T>& data,
      const T h = 1e-5,
      const int jmax = 5
    );
    */

    double update_gamma_hmc(
      const gourd::gplm_data<T>& data,
      const int integrator_steps,
      const bool update = true
    );

    double potential_energy() const;
    
    void sample_momentum_and_energy();

    void update_sigma_xi(
      const gourd::gplm_data<T>& data
    );
    
    void update_tau();

    void update_zeta();

    void set_initial_gamma(
      const gourd::gplm_data<T>& data
    );

    void set_initial_sigma(
      const gourd::gplm_data<T>& data
    );

    void set_initial_values(
      const gourd::gplm_data<T>& data
    );

    void set_initial_w(
      const gourd::gplm_data<T>& data
    );
    
  };
  // class surface_gplme_model
  /* ****************************************************************/
  
};
// namespace gourd



template< typename T >
double gourd::surface_gplme_model<T>::partial_loglik_g(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  // Can further speed up this computation?
  //  -> Already quite fast
  return mprod_trace(g, ru_ * data.xsvd_d().asDiagonal(),
		     sigma_sq_inv_) +
    -0.5 * mprod_trace(g, g * data.xsvd_d().cwiseAbs2().asDiagonal(),
		       sigma_sq_inv_);
};

/*
 * Can evaluate the full log likelihood if needed for output:
 * y' (I_n \otimes \Sigma^-1) y = tr(Y' \Sigma^-1 Y) = tr(Y Y' \Sigma^-1)
 *   = data.yssq().adjoint() * sigma_sq_inv_;
 */


template< typename T >
double gourd::surface_gplme_model<T>::partial_logprior_g(
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return -0.5 * tau_sq_inv_ * c_inv_.trqf( g );
};




template< typename T >
double gourd::surface_gplme_model<T>::partial_loglik_w(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::vector_type& w,
  const int i
) {
  return -0.5 * ( sigma_sq_inv_.asDiagonal() *
    (data.y().col(i) - conditional_mu(data, i) - w).cwiseAbs2() ).sum();
};



template< typename T >
double gourd::surface_gplme_model<T>::partial_logprior_w(
  const typename gourd::surface_gplme_model<T>::vector_type& w,
  const int i
) {
  return -0.5 * zeta_inv_.coeffRef(i) * c_inv_.q( w );
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
gourd::surface_gplme_model<T>::grad_g_log_prior(
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return -tau_sq_inv_ * c_inv_.rmul( g );
};


template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_g_log_likelihood(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) const {
  return sigma_sq_inv_.asDiagonal() * (
    ( ru_ - g * data.xsvd_d().asDiagonal() ) *
    data.xsvd_d().asDiagonal() );
};



template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_g(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::mat_type& g
) {
  return grad_g_log_likelihood(data, g) + grad_g_log_prior(g);
};






template< typename T > inline
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::conditional_mu(
  const gourd::gplm_data<T>& data,
  const int i
) const {
  assert( i >= 0 && i < data.n() &&
	  "surface_gplme_model::mu: index i out of bounds");
  return gamma_ *
    (data.xsvd_d().asDiagonal() * data.xsvd_u().row(i).adjoint());
};



template< typename T > inline
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::grad_w_log_likelihood(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::vector_type& w,
  const int i
) {
  return sigma_sq_inv_.asDiagonal() *
    ( data.y().col(i) - conditional_mu(data, i) - w );
};



template< typename T > inline
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::grad_w_log_prior(
  const typename gourd::surface_gplme_model<T>::vector_type& w,
  const int i
) {
  return -zeta_inv_.coeffRef(i) * c_inv_.rmul( w );
};





template< typename T > inline
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::grad_w(
  const gourd::gplm_data<T>& data,
  const typename gourd::surface_gplme_model<T>::vector_type& w,
  const int i
) {
  assert( i >= 0 && i < data.n() &&
	  "surface_gplme_model::grad_w: index i out of bounds");
  return grad_w_log_likelihood(data, w, i) + grad_w_log_prior(w, i);
};


/*
template< typename T > inline
typename gourd::surface_gplme_model<T>::mat_type
gourd::surface_gplme_model<T>::grad_g_fdiff(
  const gourd::gplm_data<T>& data,
  const T h,
  const int jmax
) {
  const int nloc = gamma_.rows();
  const int p = gamma_.cols();
  const int maxloc = (jmax < nloc) ? jmax : nloc;
  mat_type gradient(maxloc, p);
  T upper, lower;
  for ( int s = 0; s < maxloc; s++ ) {
    for ( int j = 0; j < p; j++ ) {
      gamma_.coeffRef(s, j) += h/2;
      upper = partial_loglik_g(data, gamma_) + partial_logprior_g(gamma_);
      gamma_.coeffRef(s, j) -= h;
      lower = partial_loglik_g(data, gamma_) + partial_logprior_g(gamma_);
      gradient.coeffRef(s, j) = (upper - lower) / h;
      // Reset gamma_
      gamma_.coeffRef(s, j) += h;
    }
  }
  return gradient;
};
*/





/* ****************************************************************/
/*
 *                               Updates
 */

template< typename T > 
double gourd::surface_gplme_model<T>::update(
  const gourd::gplm_data<T>& data,
  const int integrator_steps,
  const bool update_learning_rate
) {
  assert(integrator_steps > 0 &&
	 "surface_gplme_model: non-positive HMC integrator steps");
  update_tau();
  update_sigma_xi( data );
  update_zeta();
  const double alpha = update_gamma_hmc( data, integrator_steps );
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
  const gourd::gplm_data<T>& data,
  const int integrator_steps,
  const bool update
) {
  std::uniform_real_distribution<double> unif(0, 1);
  const T eps =
    lr_.eps( (update ? 0.9 + unif(gourd::urng()) * 0.2 : 1) );
  double log_prior_star, alpha;
  T k = 0.5;
  sample_momentum_and_energy();
  energy_initial_ = -partial_loglik_g(data, gamma_) -
    partial_logprior_g(gamma_);
  gamma_star_ = gamma_;
  momentum_ += k * eps * grad_g( data, gamma_ );
  for ( int step = 0; step < integrator_steps; step++ ) {
    k = (step == (integrator_steps - 1)) ? 0.5 : 1;
    gamma_star_.noalias() += eps * mass_.irmul( momentum_ );
    momentum_.noalias() += k * eps * grad_g( data, gamma_star_ );
  }
  // ( momentum_ *= -1 )
  log_prior_star = partial_logprior_g(gamma_star_);
  energy_proposal_ = -partial_loglik_g(data, gamma_star_) -
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
void gourd::surface_gplme_model<T>::update_sigma_xi(
  const gourd::gplm_data<T>& data
) {
  const int nloc = sigma_sq_inv_.size();
  const T shape = 0.5 * data.n() + 0.5;
  // const vector_type dsq = data.xsvd_d().cwiseAbs2();
  T rss;  // Residual sum of squares
  T sum_isig = 0;
  for ( int i = 0; i < nloc; i++ ) {
    rss = data.yssq().coeffRef(i) +
      ( gamma_.row(i) * data.xsvd_d().asDiagonal() *
	(-2 * ru_.row(i).adjoint() +
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
void gourd::surface_gplme_model<T>::update_tau() {
  const int s = gamma_.rows();
  const int p = gamma_.cols();
  const T shape = 0.5 * s * p + 1;
  const T rate = 0.5 * -log_prior_kernel_gamma_ / tau_sq_inv_ + 1;
  std::gamma_distribution<T> gam( shape, 1 / rate );
  tau_sq_inv_ = gam(gourd::urng());
};


template< typename T >
void gourd::surface_gplme_model<T>::update_zeta() {
  const int s = w_.rows();
  const T shape = 0.5 * s + 1;
  for ( int i = 0; i < w_.cols(); i++ ) {
    T rate = 0.5 * c_inv_.qf( w_.col(i) ) + 1;
    std::gamma_distribution<T> gam( shape, 1 / rate );
    zeta_inv_.coeffRef(i) = gam(gourd::urng());
  }
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
  const gourd::gplm_data<T>& data
) {
  tau_sq_inv_ = 1;
  set_initial_sigma( data );
  set_initial_gamma( data );
  /* Must be called AFTER set_initial_gamma(): */
  set_initial_w( data );
  update_tau();
  update_zeta();
  std::cout << "Initial gamma:\n"
	    << gamma_.topRows(10)
	    << "\n" << std::endl;
  std::cout << "Initial sigma:\n"
	    << sigma_sq_inv_.head(10).cwiseSqrt().cwiseInverse().adjoint()
	    << "...\n" << std::endl;
};




template< typename T > 
bool gourd::surface_gplme_model<T>::tune_initial_stepsize(
  const gourd::gplm_data<T>& data,
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
  const gourd::gplm_data<T>& data,
  const int niter,
  const int integrator_steps
) {
  assert(niter >= 0 &&
	 "surface_gplme_model: negative warmup iterations");
  assert(integrator_steps > 0 &&
	 "surface_gplme_model: non-positive integrator steps");
  const bool eps_tuned = tune_initial_stepsize(data, niter/2);
  if ( !eps_tuned && niter/2 > 0 ) {
    std::cerr << "\t*** HMC step size tuning failed\n";
  }
  for ( int i = 0; i < niter; i++ )
    update(data, integrator_steps, true);
  //
  lr_.fix();
};





template< typename T > 
void gourd::surface_gplme_model<T>::profile(
  const gourd::gplm_data<T>& data,
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
    partial_loglik_g( data, gamma_ );
    abseil::timer::stop();
    dt += abseil::timer::diff();
  }
  std::cout << "Evaluate log-likelihood  &  "
	    << (dt / (1e3 * nrep))
	    << " \\\\\n";

  
  dt = 0;
  for ( int i = 0; i < nrep; i++) {
    abseil::timer::start();
    partial_logprior_g( gamma_ );
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
void gourd::surface_gplme_model<T>::set_initial_gamma(
  const gourd::gplm_data<T>& data
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
  for ( int i = 0; i < gamma_star_.rows(); i++ ) {
    for ( int j = 0; j < z.size(); j++ ) {
      z.coeffRef(j) = normal(gourd::urng());
    }
    // gamma_.row(i) += vt_decomp.solve( di.asDiagonal() * z ).adjoint();
    gamma_.row(i) += ( data.xsvd_v() * di.asDiagonal() * z ).adjoint();
  }
  /* Set related initial values */
  gamma_star_.resize( gamma_.rows(), gamma_.cols() );
  momentum_.resize( gamma_.rows(), gamma_.cols() );
  ru_ = data.yu();
  log_prior_kernel_gamma_ = partial_logprior_g(gamma_);
  // energy_initial_ = partial_loglik_g( data, gamma_ ) +
  //   log_prior_kernel_gamma_;
};




template< typename T > 
void gourd::surface_gplme_model<T>::set_initial_sigma(
  const gourd::gplm_data<T>& data
) {
  const int nloc = data.yu().rows();
  int df = data.n() - data.xsvd().rank();
  df = (df < 1) ? 1 : df;
  T rss;
  sigma_sq_inv_.resize( nloc );
  for ( int i = 0; i < nloc; i++ ) {
    rss = data.yssq().coeffRef(i) -
      (data.yu().row(i) * data.yu().row(i).adjoint())[0];
    sigma_sq_inv_.coeffRef(i) = df / rss;
  }
  //
  xi_ = 1;
};


template< typename T >
void gourd::surface_gplme_model<T>::set_initial_w(
  const gourd::gplm_data<T>& data
) {
  using spmat_type = gourd::nngp_hess<T>::spmat_type;
  const spmat_type H = c_inv_.hessian() +    
    sigma_sq_inv_.asDiagonal();
  vector_type eta( data.nloc() );
  Eigen::SimplicialLDLT<spmat_type> ldlt;
  ldlt.compute( H );
  w_ = mat_type( data.nloc(), data.n() );
  for ( int i = 0; i < data.n(); i++ ) {
    eta = gamma_ * ( data.xsvd_d().asDiagonal() *
		     data.xsvd_u().row(i).adjoint() );
    w_.col(i) = ldlt.solve( sigma_sq_inv_.asDiagonal() *
			    (data.y().col(i) - eta) );
    ru_.noalias() -= w_.col(i) * data.xsvd_u().row(i);
  }
};

/* ****************************************************************/




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
typename gourd::surface_gplme_model<T>::vector_type
gourd::surface_gplme_model<T>::sigma() const {
  return sigma_sq_inv_.cwiseSqrt().cwiseInverse();
};


template< typename T > inline
T gourd::surface_gplme_model<T>::tau() const {
  return 1 / std::sqrt( tau_sq_inv_ );
};


template< typename T > inline
T gourd::surface_gplme_model<T>::xi() const {
  return xi_;
};
/* ****************************************************************/




template< typename T >
template< size_t D >
gourd::surface_gplme_model<T>::surface_gplme_model(
  const gourd::gplm_data<T>& data,
  const abseil::covariance_functor<T, D>* const cov,
  const T nngp_radius,
  const gourd::dist_code distance,
  const double eps0,
  const double alpha0,
  const double eps_min,
  const double g0,
  const double t0,
  const double k0
) {
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
    nngp_radius / 2,
    distance
  );
  /* Set HMC learning rate/step size */
  lr_ = abseil::learning_rate( eps0, alpha0, eps_min, g0, t0, k0 );
  /* Initialize parameters */
  set_initial_values( data );
  //
  order_update_w_.resize( data.n() );
  std::iota(order_update_w_.begin(), order_update_w_.end(), 0);
};


#endif  // _GOURD_SURFACE_GPLME_MODEL_
