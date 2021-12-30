
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
#include "gourd/pair_cifti_metric_with_gifti_surface.hpp"
#include "gourd/surface_vwise_glm.hpp"
#include "gourd/cmd/glm_command_parser.hpp"
#include "gourd/data/gplm_smoothed_sstat.hpp"





int main( const int argc, const char* argv[] ) {
#ifdef GOURD_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  using cov_type = abseil::covariance_functor<scalar_type, 3>;
  using mat_type = typename
    gourd::surface_vwise_glm<scalar_type>::mat_type;
  using vec_type = typename
    gourd::surface_vwise_glm<scalar_type>::vector_type;

  gourd::glm_command_parser input( argc, argv );
  if ( !input )  return 1;
  else if ( input.help_invoked() )  return 0;

  gourd::set_urng_seed( input.seed() );
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
    
    gourd::gplm_smoothed_sstat<scalar_type> data(
      input.covariate_file(),
      input.metric_files(),
      input.surface_file(),
      cov_ptr.get(),
      input.neighborhood(),
      input.distance_metric()
    );

    gourd::cifti_gifti_pair cgp = gourd::pair_cifti_with_gifti(
      input.metric_files()[0],
      input.surface_file()
    );

    gourd::surface_vwise_glm model( data );

    // Setup outputs
    std::vector<std::string> logids( data.p() );
    gourd::output_log ologs( input.output_basename() );
    for ( int j = 0; j < data.p(); j++ ) {
      std::ostringstream lss;
      lss << "_beta" << std::setfill('0') << std::setw(4) << j;
      logids[j] = lss.str();
      ologs.add_log( logids[j] );
    }
    mat_type beta_fm = mat_type::Zero( data.nloc(), data.p() );
    mat_type beta_sm = mat_type::Zero( data.nloc(), data.p() );
    vec_type sigma_fm = vec_type::Zero( data.nloc() );
    //

    // Run MCMC
    std::cout << "\nSampling:\n";
    const int maxit = (input.mcmc_nsamples() - 1) * input.mcmc_thin() + 1;
    for ( int i = 0; i < maxit; i++ ) {
      model.update( data );
      std::cout << "[" << (i+1) << "]\tloglik = "
		<< model.log_likelihood(data)
		<< std::endl;
      //
      if ( i % input.mcmc_thin() == 0 ) {
	mat_type beta_t = model.beta();
	beta_fm.noalias() += beta_t;
	beta_sm.noalias() += beta_t.cwiseAbs2();
	sigma_fm.noalias() += model.sigma();
	for ( int j = 0; j < beta_t.cols(); j++ ) {
	  ologs.write(
            logids[j],
	    beta_t.data() + j * data.nloc(),
	    beta_t.data() + (j+1) * data.nloc()
          );
	}
      }
    }

    // Write output images
    ::nifti_image* ref =
	gourd::nifti2::image_read( input.metric_files()[0], 0 );
    
    beta_fm /= input.mcmc_nsamples();
    beta_sm /= input.mcmc_nsamples();
    sigma_fm /= input.mcmc_nsamples();
    
    gourd::write_matrix_to_cifti(
      beta_fm, ref, cgp,
      input.output_basename() + std::string("_beta(s).dtseries.nii")
    );
    gourd::write_matrix_to_cifti(
      (beta_sm - beta_fm.cwiseAbs2()).cwiseSqrt().eval(), ref, cgp,
      input.output_basename() + std::string("_se_beta(s).dtseries.nii")
    );
    gourd::write_matrix_to_cifti(
      sigma_fm, ref, cgp,
      input.output_basename() + std::string("_sigma(s).dtseries.nii")
    );
    
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

