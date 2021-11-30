
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ansi.hpp"
#include "abseil/covariance_functors.hpp"

#include "nifti2_io.h"

#include "gourd/covariance.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/options.hpp"
#include "gourd/output.hpp"
#include "gourd/cmd/glm_command_parser.hpp"
#include "gourd/data/gplm_full_data.hpp"

#include "gourd/surface_gplmix_model.hpp"
#include "gourd/loocv.hpp"

// #ifdef GOURD_GPLMIX_NNGP_APPROX
// #include "gourd/surface_gplmix_model3.hpp"
// #include "gourd/loocv.hpp"
// #else
// #include "gourd/surface_gplmix_model.hpp"
// #endif


/* To do:
 *  - Approximation of log marginal likelihood w/out sampling?
 */



int main( const int argc, const char* argv[] ) {
#ifdef GOURD_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  using cov_type = abseil::covariance_functor<scalar_type, 3>;
  using mat_type = typename
    gourd::surface_gplmix_model<scalar_type>::mat_type;
  // using vec_type = typename
  //   gourd::surface_gplmix_model<scalar_type>::vector_type;

  gourd::glm_command_parser input( argc, argv );
  if ( !input )  return 1;
  else if ( input.help_invoked() )  return 0;

  // ::omp_set_num_threads( input.threads() );
  // Eigen::setNbThreads( input.threads() );


  try {

    std::unique_ptr<cov_type> cov_ptr;
    gourd::init_cov(
      cov_ptr,
      input.cov_function(),
      input.theta().cbegin(),
      input.theta().cend()
    );
#ifndef GOURD_GPLMIX_NNGP_APPROX
    /* For full update/HMC version, treat cov_ptr as a correlation
     * functor (set variance parameter to 1). Otherwise, retain 
     * cov_ptr's input variance parameter
     */
    cov_ptr->variance(1);
#endif
    
    gourd::gplm_full_data<scalar_type> data(
      input.metric_files(),
      input.surface_file(),
      input.covariate_file(),
      input.variance_component_indices()
    );


    gourd::surface_gplmix_model model(
      data,
      cov_ptr.get(),
      input.neighborhood(),
      input.distance_metric(),
      input.neighborhood_random_intercept(),
      input.integrator_steps(),
      input.eps(),
      input.neighborhood_mass(),
      input.metropolis_target(),
      input.eps_min()
    );

    // Setup outputs
    std::vector<std::string> logids( data.x().cols() );
    gourd::output_log ologs( input.output_basename() );
    for ( int j = 0; j < data.x().cols(); j++ ) {
      std::ostringstream lss;
      lss << "_beta" << std::setfill('0') << std::setw(3) << j;
      logids[j] = lss.str();
      ologs.add_log( logids[j] );
    }
    ologs.add_log( "_etc" );
    ologs.add_log( "_fit" );
    mat_type beta_fm = mat_type::Zero( data.nloc(), data.x().cols() );
    mat_type beta_sm = mat_type::Zero( data.nloc(), data.x().cols() );
    double llk, llk_fm = 0, llk_sm = 0;
    //

    // Run MCMC
    std::cout << "Warmup:\n";
    model.warmup( data, input.optim_maxit(), input.mcmc_burnin(),
		  input.optim_xtol() );

    if ( input.mcmc_nsamples() > 0 )  std::cout << "\nSampling:\n";
    double alpha = 0;
    const int maxit = (input.mcmc_nsamples() - 1) * input.mcmc_thin() + 1;
    for ( int i = 0; i < maxit; i++ ) {
      alpha += model.update( data, i+1 );
      //
      if ( i % input.mcmc_thin() == 0 ) {
	mat_type beta_t = model.beta();
	beta_fm += beta_t;
	beta_sm += beta_t.cwiseAbs2();
	for ( int j = 0; j < beta_t.cols(); j++ ) {
	  ologs.write(
            logids[j],
	    beta_t.data() + j * data.nloc(),
	    beta_t.data() + (j+1) * data.nloc()
          );
	}
	llk = model.log_likelihood( data );
	llk_fm += llk;  llk_sm += (llk * llk);
	ologs["_etc"] << llk << "," << model.tau() << std::endl;
      }
    }

    //
    if ( input.mcmc_nsamples() > 0 ) {
      alpha /= maxit;
      std::cout << "\t<Avg. Metropolis Rate = "
		<< std::setprecision(4) << std::fixed << alpha
		<< ">" << std::endl;
      //
      llk_fm /= input.mcmc_nsamples();
      llk_sm /= input.mcmc_nsamples();
      const double llk_var = llk_sm - llk_fm * llk_fm;
      // Print approximate log marginal likelihood:
      std::cout << "\t<log ML \u2245 "
		<< (llk_fm - 0.5 * llk_var)
		<< ">\n";
      // Print approximate number of parameters
      std::cout << "\t<Effective parameters \u2245 "
		<< (2 * llk_var)
		<< ">\n";
      //
      beta_fm /= input.mcmc_nsamples();
      beta_sm /= input.mcmc_nsamples();
      //
      // Compute DIC
      model.beta( beta_fm );
      const double dev = -2 * model.log_likelihood( data );
      const double dic = dev + 4 * llk_var;
      std::cout << "\t<DIC = " << dic << ">\n" << std::endl;
      //
      ologs["_fit"] << "Summary,Value\n"
		    << std::setprecision(6) << std::fixed
		    << "DIC," << dic << "\n"
		    << "log Marg. Likelihood,"
		    << (llk_fm - 0.5 * llk_var) << "\n"
		    << "Deviance," << dev << "\n"
		    << "Effective Parameters," << (2*llk_var) << "\n"
		    << "MH-Rate," << alpha
		    << std::endl;
    }
    else {
      beta_fm = model.beta();
      // Compute LOOCV
      const double cverr = gourd::loocv(data, model);
      std::cout << "\t<LOOCV = " << cverr << ">\n" << std::endl;
      //
      ologs["_fit"] << "Summary,Value\n"
		    << std::setprecision(6) << std::fixed
		    << "LOOCV," << cverr << "\n"
		    << "Deviance,"
		    << (-2 * model.log_likelihood(data))
		    << std::endl;
    }
    //

    // Write output images
    ::nifti_image* ref =
	gourd::nifti2::image_read( input.metric_files()[0], 0 );
    
    gourd::write_matrix_to_cifti(
      beta_fm, ref,
      input.output_basename() + std::string("_beta(s).dtseries.nii")
    );

    if ( input.mcmc_nsamples() > 0 ) {
      /* Only write variance image if variance can be estimated from
       * MCMC samples
       */
      gourd::write_matrix_to_cifti(
        (beta_sm - beta_fm.cwiseAbs2()).cwiseSqrt().eval(), ref,
	input.output_basename() + std::string("_se_beta(s).dtseries.nii")
      );
    }
    
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

