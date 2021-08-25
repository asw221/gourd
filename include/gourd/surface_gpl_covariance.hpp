
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "optim/bobyqa.h"

#include "gourd/nearest_neighbor_process.hpp"
#include "gourd/options.hpp"
#include "gourd/data/gplm_outcome_data.hpp"



#ifndef _GOURD_SURFACE_GPL_COVARIANCE_
#define _GOURD_SURFACE_GPL_COVARIANCE_


namespace gourd {
namespace sgpl {
  
#ifdef GOURD_SINGLE_PRECISION
  using real_t = float;
#else
  using real_t = double;
#endif
  using cov_t  = abseil::covariance_functor<real_t, 3>;
  using data_t = gourd::gplm_outcome_data<real_t>;
  using param_t  = typename cov_t::param_type;
  using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;


  enum class optim_code {
    unknown,
    success             = BOBYQA_SUCCESS,
    interpolation_error = BOBYQA_BAD_NPT,
    boundary_error      = BOBYQA_TOO_CLOSE,
    precision_error     = BOBYQA_ROUNDING_ERRORS,
    maxit_error         = BOBYQA_TOO_MANY_EVALUATIONS,
    trust_region_error  = BOBYQA_STEP_FAILED
  };
  
  
  struct data_pack {
    cov_t  * cov_ptr;
    data_t * data_ptr;
    vector_t marginal_var;
    real_t   neighborhood_radius;
    gourd::dist_code distance_metric;
  };

  struct optim_output {
    gourd::sgpl::optim_code code;
    std::vector<real_t>     theta;
    real_t                  objective;
  };


  

  /*! Negative marginal log-likelihood for surface GP linear model 
   * 
   * Defined to work conveniently with external optimization 
   * routines.
   */
  double nmloglik( const long n, const double* x, void* data );


  /*! Optimize covariance parameters 
   *
   * @param n
   *   Number of parameters to optimize
   *
   * @param maxit
   *   Maximum number of iterations
   *
   * @param tol
   *   Numerical tolerance
   *
   * @param verbosity
   *   Integer from 0 (no printout), ..., 3 (max printout)
   */
  optim_output optimize_cov_params(
    gourd::sgpl::data_pack& d,
    const int n = 3,
    const int maxit = 500,
    const double tol = 1e-6,
    const int verbosity = 0
  );
  

}  // namespace sgpl
}  // namespace gourd



/* Stream insertion operator for \c optim_code */
std::ostream& operator<<( std::ostream& os, gourd::sgpl::optim_code c );


double gourd::sgpl::nmloglik(
  const long n,
  const double* x,
  void* data
) {
  gourd::sgpl::data_pack* dptr =
    static_cast< gourd::sgpl::data_pack* >( data );
  const int npar = dptr->cov_ptr->param().size();
  gourd::sgpl::vector_t nugget = dptr->marginal_var;
  nugget.noalias() -=
    gourd::sgpl::vector_t::Constant(nugget.size(), x[0]);
  // std::vector<gourd::sgpl::real_t> th( x, x + npar );
  // param_t theta( th.begin(), th.end() );
  param_t theta( x, x + npar );
  dptr->cov_ptr->param( theta );
  gourd::nnp_hess<gourd::sgpl::real_t> vinv(
    dptr->data_ptr->coordinates(),
    dptr->cov_ptr,
    nugget,
    dptr->neighborhood_radius,
    dptr->distance_metric,
    false
  );
  double objective = vinv.ldet();
  for ( int i = 0; i < dptr->data_ptr->n(); i++ ) {
    objective += 0.5 * vinv.qf( dptr->data_ptr->y().col(i) );
  }
  return objective;
};





gourd::sgpl::optim_output
gourd::sgpl::optimize_cov_params(
  gourd::sgpl::data_pack& d,
  const int n,
  const int maxit,
  const double tol,
  const int verbosity
) {
  assert( n > 0 && n <= 3 &&
	  "optimize_cov_params: n should be on [1, 3]" );
  assert( maxit >= 1 && "optimize_cov_params: maxit < 1" );
  assert( tol > 0 && tol < 1 &&
	  "optimize_cov_params: tol should be on (0, 1)");
  assert( verbosity >= 0 && verbosity <= 3 &&
	  "optimize_cov_params: verbosity should be on [0, 3]" );
  /* Set starting values */
  /* Size of parameter */
  const int npar = d.cov_ptr->param().size();
  /* Interpolation conditions */
  const int npt = (n + 1) * (n + 2) / 2;
  /* Size of working space */
  const int nw = (npt + 5) * (npt + n) + 3 * n * (n + 5) / 2;
  /* Parameter & bounds */
  std::vector<double> x(npar);
  std::vector<double> ub(npar);
  std::vector<double> lb(npar);
  for ( size_t i = 0; i < x.size(); i++ ) {
    x[i]  = static_cast<double>( d.cov_ptr->param()[i] );
    lb[i] = static_cast<double>( d.cov_ptr->param_lower_bounds()[i] );
    ub[i] = static_cast<double>( d.cov_ptr->param_upper_bounds()[i] );
    /* Adjust lower/upper bounds (mostly a just-in-case) */
    lb[i] += tol / 10;
    ub[i] -= tol / 10;
    if ( lb[i] > ub[i] ) {
      double temp = lb[i];
      lb[i] = ub[i];
      ub[i] = temp;
    }
    if ( lb[i] == ub[i] ) {
      ub[i] += tol / 20;
    }
  }
  ub[0] = d.marginal_var.minCoeff();
  if ( ub[0] <= tol ) {
    throw std::domain_error(
      "optimize_cov_params: Data marginal variance <= tol");
  }
  /* Compute approx trust region radius
   *   - Bobyqa throws an error if any (ub[i] - lb[i])/2 > rhobeg
   *   - rhobeg is supposed to be about one 10th the above size
   */
  double rhobeg = ub[0] - lb[0];
  for ( int i = 1; i < n; i++ ) {
    rhobeg = std::min( rhobeg, ub[i] - lb[i] );
  }
  rhobeg /= 20;
  /* Allocate memory to a working space */
  std::vector<double> work(nw);
  /* Return value */
  gourd::sgpl::optim_output out;
  //
  const int code = ::bobyqa(
    n, npt, &gourd::sgpl::nmloglik, (void*)(&d),
    x.data(), lb.data(), ub.data(),
    rhobeg, std::min(rhobeg, tol),
    verbosity, maxit, work.data()
  );
  out.code = static_cast<gourd::sgpl::optim_code>( code );
  out.theta = std::vector<gourd::sgpl::real_t>( x.begin(), x.end() );
  out.objective = work[0];
  return out;
};




std::ostream& operator<<(
  std::ostream& os,
  gourd::sgpl::optim_code c
) {
  using code = gourd::sgpl::optim_code;
  switch (c) {
  case code::interpolation_error : os << "interpolation error"; break;
  case code::boundary_error      : os << "boundary error"; break;
  case code::precision_error     : os << "precision error"; break;
  case code::maxit_error         : os << "maxit reached"; break;
  case code::trust_region_error  : os << "trust region error"; break;
  case code::unknown             : os << "unknown error"; break;
  default                        : os << "[]"; break;
  };
  return os;
};


#endif  // _GOURD_SURFACE_GPL_COVARIANCE_

