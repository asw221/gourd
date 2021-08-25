
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>



#ifndef _ABSEIL_MCMC_LEARNING_RATE_
#define _ABSEIL_MCMC_LEARNING_RATE_



namespace abseil {


  /* ****************************************************************/
  /*! Learning rate parameter for gradient-based MCMC
   * 
   * Implements the dual-averaging method of
   * <a href="http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf">Hoffman and Gelman, 2014</a>
   * to automatically tune the step size, \c eps(). The value returned 
   * by \c eps() has an enforced lower bound set to the value of 
   * \c eps_min().
   */
  class learning_rate {
  public:

    /*! (constructor)
     * @param starting_eps  Initial learning rate
     * @param alpha_target  Target Metropolis Hastings rate
     * @param eps_min       Enforced minimum learning rate
     * @param gamma         Tuning parameter. See Hoffman & Gelman
     * @param t0            Tuning parameter. See Hoffman & Gelman
     * @param kappa         Tuning parameter. See Hoffman & Gelman
     */
    explicit learning_rate(
      const double starting_eps = 0.1,
      const double alpha_target = 0.65,
      const double eps_min = 1e-5,
      const double gamma = 0.05,
      const double t0 = 10,
      const double kappa = 0.75
    );

    // learning_rate() : learning_rate(0.1)
    // { ; }

    learning_rate( const learning_rate& other );

    learning_rate& operator=( const learning_rate& other );

    /*! Adjust initial value of \c eps
     * 
     * Computes simple updates to \c eps until the input
     * Metropolis Hastings rate crosses 0.5; returns \c false
     * when this criterion is met (indicating further calls to
     * \c adust_initial_value are not necessary)
     */
    bool adjust_initial_value( const double alpha );

    /*! Return step size/learning rate */
    double eps( const double scale = 1 ) const;
    operator double() const { return eps(); };
    
    /*! Return enforced minimum value of \c eps() */
    double eps_min() const;

    /*! Update based on Metropolis Hastings rate */
    void adapt( const double alpha );

    /*! Fix values against further adaptation */
    void fix();

    /*! Reset to near initial state */
    void reset();
    
    
  private:
    bool fixed_;  /* Limits scope of adaptation */
    
    double eps_;           /* Learning rate */
    double eps_lb_;        /* Minimum value of \c eps */
    double alpha_target_;  /* Target Metropolis Hastings rate */
    double t_;   /* Count of calls to \c adapt */
    

    /* Dual average update parameters */
    double gamma_;
    double t0_;
    double kappa_;
    
    /* Internally managed */
    double eps_bar_;
    double eps_target_;
    double h_bar_;
  };
  // class learning_rate
  /* ****************************************************************/

  
}
// namespace abseil




abseil::learning_rate::learning_rate(
  const double starting_eps,
  const double alpha_target,
  const double eps_min,
  const double gamma,
  const double t0,
  const double kappa
) {
  assert( starting_eps > 0 && "learning_rate: starting_eps <= 0" );
  assert( alpha_target > 0 && alpha_target < 1 &&
	  "learning_rate: alpha_target should be on (0, 1)" );
  assert( gamma > 0 && gamma < 1 &&
	  "learning_rate: gamma should be on (0, 1)" );
  assert( t0 >= 0 && "learning_rate: t0 < 0" );
  assert( kappa > 0 && kappa < 1 &&
	  "learning_rate: kappa should be on (0, 1)");

  fixed_ = false;
  
  alpha_target_ = alpha_target;
  
  eps_ = std::max(starting_eps, eps_min);
  eps_bar_ = 1;
  eps_lb_ = eps_min;
  eps_target_ = std::log(10 * eps_);
  gamma_ = gamma;
  h_bar_ = 0;
  t0_ = t0;
  kappa_ = kappa;
  
  t_ = 0;
};


abseil::learning_rate::learning_rate(
  const abseil::learning_rate& other
) {
  *this = other;
};


abseil::learning_rate& abseil::learning_rate::operator=(
  const abseil::learning_rate& other
) {
  if ( &other == this ) return *this;
  eps_ = other.eps_;
  eps_lb_ = other.eps_lb_;
  alpha_target_ = other.alpha_target_;
  t_ = other.t_;

  /* Dual average update parameters */
  gamma_ = other.gamma_;
  t0_ = other.t0_;
  kappa_ = other.kappa_;
    
  /* Internally managed */
  fixed_ = other.fixed_;
  eps_bar_ = other.eps_bar_;
  eps_target_ = other.eps_target_;
  h_bar_ = other.h_bar_;
  return *this;
};




double abseil::learning_rate::eps( const double scale ) const {
  return std::max(eps_ * scale, eps_lb_);
};



double abseil::learning_rate::eps_min() const {
  return eps_lb_;
};




bool abseil::learning_rate::adjust_initial_value(
  const double alpha
) {
  static const double a = 2 * int(alpha > 0.5) - 1;
  if ( !fixed_ ) {

    const double mhr = std::min(std::max(alpha, 0.0), 1.0);
    const bool more_adjustment_needed =
      std::pow(mhr, a) > std::pow(2.0, -a);
    if ( more_adjustment_needed ) {
      eps_ *= std::pow(2.0, a);
    }
    eps_target_ = std::log(10 * eps_);
    
    return more_adjustment_needed;
  }
  return false;
};



void abseil::learning_rate::adapt(
  const double alpha
) {
  if ( !fixed_ ) {
    const double mhr = std::min(std::max(alpha, 0.0), 1.0);
    h_bar_ = (1 - 1 / (t_ + t0_)) * h_bar_ +
      (alpha_target_ - mhr) / (t_ + t0_);
    eps_ =
      std::exp(eps_target_ - std::sqrt(t_) *
	       h_bar_ / gamma_);
    eps_ = std::max(eps_, eps_lb_);
    eps_bar_ =
      std::exp(std::pow(t_, -kappa_) *
	       std::log(eps_) +
	       (1 - std::pow(t_, -kappa_)) *
	       std::log(eps_bar_));
  }
  t_++;
};


void abseil::learning_rate::fix() {
  fixed_ = true; 
  if ( eps_bar_ > eps_lb_ )
    eps_ = eps_bar_;
};



void abseil::learning_rate::reset() {
  t_ = 0;
  fixed_ = false;
};



#endif  // _ABSEIL_MCMC_LEARNING_RATE_
