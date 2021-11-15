
#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "ansi.hpp"
#include "abseil/covariance_functors.hpp"

#include "gourd/covariance.hpp"
#include "gourd/options.hpp"
#include "gourd/surface_gpl_covariance.hpp"
#include "gourd/cmd/covest_command_parser.hpp"
#include "gourd/data/gplm_outcome_data.hpp"




int main( const int argc, const char* argv[] ) {
#ifdef GOURD_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  using cov_type = abseil::covariance_functor<scalar_type, 3>;
  using vector_type = typename gourd::gplm_outcome_data
    <scalar_type>::vector_type;
  
  gourd::covest_command_parser input( argc, argv );
  if ( !input )  return 1;
  else if ( input.help_invoked() )  return 0;


  try {
    std::unique_ptr<cov_type> cov_ptr;
    gourd::init_cov(
      cov_ptr,
      input.cov_function(),
      input.theta().cbegin(),
      input.theta().cend()
    );
    
    gourd::gplm_outcome_data<scalar_type> data(
      input.metric_files(),
      input.surface_file()
    );
    data.center();

    gourd::sgpl::data_pack pack;
    pack.cov_ptr  = cov_ptr.get();
    pack.data_ptr = &data;
    pack.neighborhood_radius = input.neighborhood();
    pack.distance_metric = input.distance_metric();
    /* Initialize pack.marginal_var:
     * If given more than one patient's data, estimate marginal variance
     * voxelwise; otherwise, for singe patient analysis, estimate
     * the marginal variance as a scalar across the entire image
     */
    if ( data.n() > 1 ) {
      pack.marginal_var = data.y().cwiseAbs2().rowwise().mean() -
	data.y().rowwise().mean().cwiseAbs2();
    }
    else {
      const scalar_type ybar  = data.y().mean();
      const scalar_type var_y = data.y().cwiseAbs2().mean() -
	ybar * ybar;
      pack.marginal_var = vector_type::Constant( data.nloc(), var_y );
    }

    /* Optimization */
    gourd::sgpl::optim_output opt = gourd::sgpl::optimize_cov_params(
      pack, input.which_params(),
      input.maxit(), input.tol(), input.print_level()
    );

    /* Output formatting */
    if ( opt.code != gourd::sgpl::optim_code::success ) {
      std::cerr << ansi::bold << ansi::magenta
		<< "Optimizer reported error state: "
		<< opt.code << ansi::reset << std::endl;
    }

    std::cout << "\nOptimal " << cov_ptr->param() << "\n"
	      << "  -> FWHM: " << cov_ptr->fwhm() << " mm\n"
	      << "  -> Log-likelihood: " << -opt.objective << "\n\n"
	      << "To pass the result to another program, use:\n"
	      << "\t";
    switch (input.cov_function()) {
      case gourd::cov_code::rbf    : std::cout << "-rbf"; break;
      case gourd::cov_code::rq     : std::cout << "-rq"; break;
      case gourd::cov_code::matern : std::cout << "--matern"; break;
    }
    std::cout << " --theta ";
    for ( auto th : opt.theta )  std::cout << th << " ";
    std::cout << "\n" << std::endl;

    /* ~~ Fin ~~ */
  }
  catch ( const std::exception& ex ) {
    std::cerr << ansi::bold << ansi::magenta
	      << "*** Exception caught:\n"
	      << ansi::reset
	      << ex.what()
	      << std::endl;
    return 1;
  }
  catch ( ... ) {
    std::cerr << ansi::bold << ansi::magenta
	      << "*** Program error (unknown cause)"
	      << ansi::reset
	      << std::endl;
    return 1;
  }
};
