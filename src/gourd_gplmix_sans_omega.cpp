
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
#include "gourd/surface_gplmix_model.hpp"
#include "gourd/cmd/glm_command_parser.hpp"
#include "gourd/data/gplmix_sstat.hpp"





int main( const int argc, const char* argv[] ) {
#ifdef GOURD_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  using cov_type = abseil::covariance_functor<scalar_type, 3>;
  using mat_type = typename
    gourd::surface_gplmix_model<scalar_type>::mat_type;
  using vec_type = typename
    gourd::surface_gplmix_model<scalar_type>::vector_type;

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
    
    gourd::gplmix_sstat<scalar_type> data(
      input.covariate_file(),
      input.metric_files(),
      input.surface_file(),
      input.variance_component_indices()
    );

    // std::cout << "d = " << data.xsvd_d().adjoint() << "\n"
    // 	      << "U = " << data.xsvd_u().topRows(6) << "\n"
    // 	      << "V = " << data.xsvd_v() << "\n"
    // 	      << std::endl;

    gourd::surface_gplmix_model model(
      data,
      cov_ptr.get(),
      input.neighborhood(),
      input.distance_metric(),
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
      lss << "beta" << std::setfill('0') << std::setw(3) << j;
      logids[j] = lss.str();
      ologs.add_log( logids[j] );
    }
    mat_type beta_fm = mat_type::Zero( data.nloc(), data.x().cols() );
    mat_type beta_sm = mat_type::Zero( data.nloc(), data.x().cols() );
    vec_type sigma_fm = vec_type::Zero( data.nloc() );
    //

    // Run MCMC
    model.warmup( data, input.mcmc_burnin() );

    double alpha = 0;
    const int maxit = (input.mcmc_nsamples() - 1) * input.mcmc_thin() + 1;
    for ( int i = 0; i < maxit; i++ ) {
      alpha += model.update( data );
      //
      if ( i % input.mcmc_thin() == 0 ) {
	mat_type beta_t = model.beta();
	beta_fm += beta_t;
	beta_sm += beta_t.cwiseAbs2();
	sigma_fm += model.sigma();
	for ( int j = 0; j < beta_t.cols(); j++ ) {
	  ologs.write(
            logids[j],
	    beta_t.data() + j * data.nloc(),
	    beta_t.data() + (j+1) * data.nloc()
          );
	}
      }
    }

    //
    alpha /= maxit;
    std::cout << "\t<Avg. Metropolis Rate = "
	      << std::setprecision(4) << std::fixed << alpha
	      << ">" << std::endl;

    // Write output images
    ::nifti_image* ref =
	gourd::nifti2::image_read( input.metric_files()[0], 0 );
    
    beta_fm /= input.mcmc_nsamples();
    beta_sm /= input.mcmc_nsamples();
    sigma_fm /= input.mcmc_nsamples();
    
    gourd::write_matrix_to_cifti(
      beta_fm, ref,
      input.output_basename() + std::string("beta(s).dtseries.nii")
    );
    gourd::write_matrix_to_cifti(
      (beta_sm - beta_fm.cwiseAbs2()).cwiseSqrt().eval(), ref,
      input.output_basename() + std::string("se_beta(s).dtseries.nii")
    );
    gourd::write_matrix_to_cifti(
      sigma_fm, ref,
      input.output_basename() + std::string("sigma(s).dtseries.nii")
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

