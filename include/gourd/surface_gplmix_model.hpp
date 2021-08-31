
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/mcmc/learning_rate.hpp"

#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/rng.hpp"
#include "gourd/data/gplm_full_data.hpp"

// Profiling
#include "abseil/timer.hpp"
//


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

    /*
     * Needs to construct:
     *  - nnp_hess( vector<cartesian_coordinate>,
     *              covariance_functor, T radius, dist_code )
     *  - learning_rate( double eps0, double target_mh )
     */
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
      const double mass_rad = 4,   /*!< HMC mass matrix n'hood radius */
      const double alpha0 = 0.65,  /*!< HMC target Metropolis-Hastings rate */
      const double eps_min = 1e-5, /*!< HMC minimum step size */
      const double g0 = 0.05,      /*!< HMC dual-averaging tuning parameter */
      const double t0 = 10,        /*!< HMC dual-averaging tuning parameter */
      const double k0 = 0.75       /*!< HMC dual-averaging tuning parameter */
    );

    surface_gplmix_model() = default;
    
    T sigma() const;
    const mat_type& beta() const;
    vector_type tau() const;
    
    double update(
      const gourd::gplm_full_data<T>& data,
      const int monitor = 0,
      const bool update_learning_rate = false
    );
    
    void update_error_terms( const gourd::gplm_full_data<T>& data );
    // void update_eta();
    // void update_tau();

    bool tune_initial_stepsize(
      const gourd::gplm_full_data<T>& data,
      const int maxit = 100
    );

    void warmup(
      const gourd::gplm_full_data<T>& data,
      const int niter
    );

    void profile(
      const gourd::gplm_full_data<T>& data,
      const int nrep = 10
    );
    
    
  protected:

    /* Model parameters */
    mat_type beta_;
    mat_type gamma_;            /* gamma = rotated beta: B V */
    mat_type gamma_star_;
    mat_type omega_;            /* Random spatial intercepts */
    T sigma_sq_inv_;
    T tau_sq_inv_;
    T eta_sq_inv_;
    // T xi_;
    gourd::nnp_hess<T> c_inv_;
    gourd::nnp_hess<T> c_tilde_inv_;  /* Inverse correlation of the 
				       * \omega_i: 
				       * smaller neighborhoods */

    /* Parameters related to updates */
    mat_type yu_;
    mat_type momentum_;
    mat_type vt_;                /* X = U D ~Vt~ */
    double energy_initial_;      /* Partial log posterior */
    double energy_momentum_;
    double energy_proposal_;
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

    void set_initial_omega(
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
double gourd::surface_gplmix_model<T>::partial_loglik(
  const gourd::gplm_full_data<T>& data,
  const typename gourd::surface_gplmix_model<T>::mat_type& g
) const {
  const mat_type ud = data.xsvd_u() * data.xsvd_d().asDiagonal();
  T lk = 0;
  for ( int i = 0; i < data.n(); i++ ) {
    vector_type resid = data.y(i) -
      (omega_.col(i) + g * ud.row(i).adjoint());
    lk += resid.squaredNorm();
    // lk += ( resid.adjoint() * sigma_sq_inv_.asDiagonal() *
    // 	    resid ).coeff(0);
  }
  return -0.5 * sigma_sq_inv_ * lk;
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


/* Compute tr(a' diag(v) b) */
template< typename T >
double gourd::surface_gplmix_model<T>::mprod_trace(
  const typename gourd::surface_gplmix_model<T>::mat_type& a,
  const typename gourd::surface_gplmix_model<T>::mat_type& b,
  const typename gourd::surface_gplmix_model<T>::vector_type& v
) const {
  assert( a.cols() == b.cols() &&
	  a.rows() == b.rows() &&
	  a.rows() == v.size() &&
	  "surface_gplmix_model:matrix trace: dimensions must agree" );
  double trace = 0;
  for ( int j = 0; j < a.cols(); j++ )
    trace += static_cast<double>(
      (a.col(j).adjoint() * v.asDiagonal() * b.col(j)).coeff(0) );
  return trace;
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
  return sigma_sq_inv_ *
    ( data.yu() - omega_ * data.xsvd_u() -
      g * data.xsvd_d().asDiagonal() ) * data.xsvd_d().asDiagonal();
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
  update_error_terms( data );
  const double alpha = update_gamma_hmc( data, leapfrog_steps_ );
  if ( monitor > 0 ) {
    std::cout << "[" << monitor << "]\t\u03b1 = " << alpha
	      << "\tloglik = " << partial_loglik(data, gamma_)
	      << "\t\u03b5 = " << lr_
	      << std::endl;
  }
  //
  // std::cout << "B =\n" << beta_.topRows(10) << "\n";
  // std::cout << "W =\n" << omega_.topLeftCorner(10, 4) << "\n";
  // std::cout << "\u03c4\u00b2 = " << tau_sq_inv_ << "\n";
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
    beta_ = gamma_ * vt_;
  }
  return (alpha > 1) ? 1 : alpha;
};
  // \epsilon <---> \u03b5





template< typename T >
void gourd::surface_gplmix_model<T>::update_error_terms(
  const gourd::gplm_full_data<T>& data
) {
  using gamma_par_t = typename std::gamma_distribution<T>::param_type;
  using spmat_t = Eigen::SparseMatrix<T>;
  // First update the \omega_i
  spmat_t vinv = eta_sq_inv_ * c_tilde_inv_.hessian();
  vinv += vector_type::Constant(vinv.rows(), sigma_sq_inv_)
    .asDiagonal();
  Eigen::SimplicialLDLT<spmat_t> ldlv(vinv);
  // vector_type rss = vector_type::Zero( data.nloc() );
  T rss = 0;
  T omega_qf = 0;
  for ( int i = 0; i < data.n(); i++ ) {
    std::normal_distribution<T> normal(0, 1);
    vector_type resid = data.y(i) - beta_ * data.x().row(i).adjoint();
    vector_type z(resid.size());
    for ( int j = 0; j < z.size(); j++ ) {
      z.coeffRef(j) = normal(gourd::urng());
    }
    omega_.col(i) = ldlv.solve(
      sigma_sq_inv_ * resid +
      ldlv.vectorD().cwiseSqrt().asDiagonal() * (ldlv.matrixU() * z)
    );
    /* V^-1 = L D L' 
     * Want: V^1/2 z
     *   solve(V^-1, x) = V^-1/2 z
     *   V^-1/2 = D^1/2 L'
     */
    // rss.noalias() += ( resid - omega_.col(i) ).cwiseAbs2();
    rss += (resid - omega_.col(i)).squaredNorm();
    omega_qf += static_cast<T>( c_tilde_inv_.qf(omega_.col(i)) );
  }
  // \sigma^-2
  const T shape_sigma = 0.5 + 0.5 * data.n() * data.nloc();
  const T rate_sigma = 0.5 + 0.5 * rss;
  std::gamma_distribution<T> gamma(shape_sigma, 1/rate_sigma);
  sigma_sq_inv_ = gamma(gourd::urng());
  //
  // \eta^-2
  T gamma_qf = 0;
  for ( int j = 0; j < gamma_.cols(); j++ ) {
    gamma_qf += c_inv_.qf(gamma_.col(j));
  }
  const T shape_eta = 0.5 + 0.5 * data.nloc() * (data.n() + data.p());
  const T rate_eta = 0.5 + 0.5 * (omega_qf + gamma_qf * tau_sq_inv_);
  gamma.param( gamma_par_t(shape_eta, 1/rate_eta) );
  eta_sq_inv_ = gamma(gourd::urng());
  // \tau^-2
  const T shape_tau = 0.5 + 0.5 * data.nloc() * data.p();
  const T rate_tau = 0.5 + 0.5 * gamma_qf * eta_sq_inv_;
  gamma.param( gamma_par_t(shape_tau, 1/rate_tau) );
  tau_sq_inv_ = gamma(gourd::urng());
  // T sumsig = 0;
  // for ( int s = 0; s < rss.size(); s++ ) {
  //   const T shape_sigma = 0.5 * data.n() + 0.5;
  //   const T rate_sigma = xi_ + rss.coeffRef(s) / 2;
  //   std::gamma_distribution<T> gamma(shape_sigma, 1/rate_sigma);
  //   sigma_sq_inv_.coeffRef(s) = gamma(gourd::urng());
  //   sumsig += sigma_sq_inv_.coeffRef(s);
  // }
  // // \xi
  // const T shape_xi = 0.5 * data.nloc() + 0.5;
  // const T rate_xi = 1 + sumsig;
  // std::gamma_distribution<T> gamma(shape_xi, 1/rate_xi);
  // xi_ = gamma(gourd::urng());
};




// template< typename T > 
// void gourd::surface_gplmix_model<T>::update_tau() {
//   const T shape = 0.5 * gamma_.rows() * gamma_.cols() + 1;
//   const T rate  = 1 - log_prior_kernel_gamma_ /
//     (tau_sq_inv_ * eta_sq_inv_);
//   std::gamma_distribution<T> gamma( shape, 1/rate );
//   tau_sq_inv_ = gamma(gourd::urng());
// };


// template< typename T > 
// void gourd::surface_gplmix_model<T>::update_eta() {
//   const T shape = 1 + 0.5 * gamma_.rows() *
//     (gamma_.cols() + omega_.cols());
//   const T rate = 1 -
//     log_prior_kernel_gamma_ / (tau_sq_inv_ * eta_sq_inv_) -
//     log_prior_kernel_omega_ / eta_sq_inv_;
//   std::gamma_distribution<T> gamma( shape, 1/rate );
//   eta_sq_inv_ = gamma(gourd::urng());
// };
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
  momentum_ = mass_.hprod(momentum_).eval();
  /* ^^ Inefficient? Not really. Appears to be very marginally faster 
   * than preallocating the memory and copying into momentum_
   */
};



template< typename T > 
double gourd::surface_gplmix_model<T>::potential_energy() const {
  return 0.5 * mass_.triqf(momentum_);
};






/* ****************************************************************/
/*
 *                           Initial values 
 */
template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_values(
  const gourd::gplm_full_data<T>& data
) {
  tau_sq_inv_ = 1; eta_sq_inv_ = 1;
  set_initial_sigma( data );
  set_initial_gamma( data );
  set_initial_omega( data );
  // update_tau(); /* Must be called AFTER set_initial_gamma() */
  //
#ifndef NDEBUG
  std::cout << "Initial gamma:\n" << gamma_.topRows(10)
  	    << "\n" << std::endl;
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
  const gourd::gplm_full_data<T>& data,
  const int niter
) {
  assert(niter >= 0 &&
	 "surface_gplmix_model: negative warmup iterations");
  const bool eps_tuned = tune_initial_stepsize(data, 20);
  if ( !eps_tuned ) {
    std::cerr << "\t*** HMC step size tuning failed\n";
  }
  for ( int i = 0; i < niter; i++ )
    update(data, i+1, true);
  lr_.fix();
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
  // spmat_t vi = c_inv_.hessian();
  // vi += vector_type::Constant(data.nloc(), sigma_sq_inv_)
  //   .asDiagonal();
  // Eigen::SimplicialLDLT<spmat_t> ldl(vi);
  // gamma_ = ldl.solve( data.yu() * di.asDiagonal() );
  //
  gamma_ = data.yu() * di.asDiagonal();
  //
  // gamma_ = mat_type::Zero( data.nloc(), data.p() );
  vt_ = data.xsvd_v().adjoint();
  /* Jitter initial values */
  // const Eigen::LDLT<mat_type> vt_decomp = vt_.ldlt();
  T tau = std::sqrt(
    (gamma_.cwiseAbs2().colwise().mean() -
     gamma_.colwise().mean().cwiseAbs2()).minCoeff()
   );
  // T tau = 1;
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
  beta_ = gamma_ * vt_;
  gamma_star_.resize( gamma_.rows(), gamma_.cols() );
  momentum_.resize( gamma_.rows(), gamma_.cols() );
};



template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_omega(
  const gourd::gplm_full_data<T>& data
) {
  omega_ = mat_type::Zero( data.nloc(), data.n() );
};


template< typename T > 
void gourd::surface_gplmix_model<T>::set_initial_sigma(
  const gourd::gplm_full_data<T>& data
) {
  const int ns = data.n() * data.nloc();
  T first = 0, second = 0;  // moments
  for ( int i = 0; i < data.n(); i++ ) {
    first += data.y(i).sum();
    second += data.y(i).cwiseAbs2().sum();
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
T gourd::surface_gplmix_model<T>::sigma() const {
  return std::sqrt(1 / sigma_sq_inv_);
};


template< typename T > inline
typename gourd::surface_gplmix_model<T>::vector_type
gourd::surface_gplmix_model<T>::tau() const {
  return tau_sq_inv_.cwiseInverse().cwiseSqrt();
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
  /* Set Hessian */
  c_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov,
    nngp_radius,
    distance
  );
  /* Random intercept Hessian */
  c_tilde_inv_ = gourd::nnp_hess<T>(
    data.coordinates(),
    cov,
    mixef_rad,
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
