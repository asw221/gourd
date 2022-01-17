
#include <chrono>
#include <cstdio>  // std::remove
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "abseil/utilities/csv_reader.hpp"

#include "gourd/options.hpp"  // gourd::cov_code
#include "gourd/cmd/basic_command_parser.hpp"
#include "gourd/utilities/parse_indices_text.hpp"
#include "gourd/utilities/ragged_array.hpp"


#ifndef _GOURD_GLM_COMMAND_PARSER_
#define _GOURD_GLM_COMMAND_PARSER_


namespace gourd {

  /* ****************************************************************/
  /*! Parse command line input for gourd's regression models
   */
  class glm_command_parser :
    public gourd::basic_command_parser {
  public:
    glm_command_parser(
      const int argc,
      const char** argv,
      const bool warn_unrecognized = true
    );

    void show_help() const override;
    void show_usage() const override;

    bool profile_computation() const;

    double eps() const;
    double eps_min() const;
    double metropolis_target() const;
    double neighborhood() const;
    double neighborhood_random_intercept() const;
    double neighborhood_mass() const;
    double optim_xtol() const;
    double target_mh_rate() const;

    gourd::cov_code cov_function() const;
    gourd::dist_code distance_metric() const;

    int integrator_steps() const;
    int mcmc_burnin() const;
    int mcmc_nsamples() const;
    int mcmc_thin() const;
    int optim_maxit() const;
    int threads() const;
    
    unsigned seed() const;
    std::string covariate_file() const;
    std::string output_basename() const;
    std::string subset_file() const;
    std::string surface_file() const;
    const std::vector<double>& theta() const;
    const std::vector<std::string>& metric_files() const;

    gourd::ragged_array<int> variance_component_indices() const;

    
  private:
    gourd::cov_code  covar_;
    gourd::dist_code dist_;
    bool profile_;
    double eps_;
    double epsmin_;
    double nhood_;
    double nhood_rint_;
    double nhood_mass_;
    double target_mh_;
    double xtol_;
    int burnin_;
    int maxit_;
    int nsamples_;
    int steps_;
    int thin_;
    int threads_;
    std::string covariate_file_;
    std::string varcomp_indices_;
    std::string output_basename_;
    std::string subset_file_;
    std::string surface_file_;
    unsigned seed_;
    std::vector<double> theta_;
    std::vector<std::string> metric_files_;

    void subset_metric_files_();
  };
  // class glm_command_parser
  /* ****************************************************************/

};
// namespace gourd



void gourd::glm_command_parser::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "  --surface     file/path  REQURED. GIFTI surface file \n"
	    << "\n"
	    << "  --covariates  file/path  Mean model covariates \n"
	    << "  --seed          int      URNG seed \n"
	    << "  --subset      file/path  File with tokens to select metric files \n"
	    << "  --theta        float...  Spatial cov. function parameters \n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << std::endl;
};


void gourd::glm_command_parser::show_usage() const {
  std::cerr << "Usage:\n"
	    << "\t" << caller()
	    << "  path/to/data*.nii --surface path/to/surf.gii <options>\n"
	    << std::endl;
};





gourd::glm_command_parser::glm_command_parser(
  const int argc,
  const char *argv[],
  const bool warn_unrecognized
) :
  gourd::basic_command_parser( argc, argv )
{
  /* --- Default Values ------------------------------------------- */
  covar_  = gourd::cov_code::rbf;
  dist_   = gourd::dist_code::great_circle;
  
  const auto call_time =
    std::chrono::system_clock::now().time_since_epoch();
  seed_ = (unsigned)std::chrono::duration_cast
    <std::chrono::seconds>(call_time).count();

  eps_    = 0.1;
  epsmin_ = 1e-5;
  nhood_ = 6;
  nhood_rint_ = 4;
  nhood_mass_ = 0;
  target_mh_  = 0.65;

  steps_    = 10;
  burnin_   = 1000;
  maxit_    = 500;
  nsamples_ = 1000;
  thin_     = 1;
  threads_  = 0;

  output_basename_ = "";
  profile_ = false;
  

  /* Parameters that do not have default values:
   *
   *   - covariates_file_;
   *   - surface_file_;
   *   - theta_;
   *   - metric_files_;
   */
  /* -------------------------------------------------------------- */

  
  if ( argc >= 2 ) {
    for ( int i = 1; i < argc; i++ ) {
      
      std::string arg = argv[i];

      if ( arg == "-h" || arg == "--help" ) {
	this->status_ = call_status::help;
	break;
      }
      else if ( arg == "--covariates" || arg == "-x" ||
		arg == "--covariate" ) {
	process_file_argument( argc, argv, i, covariate_file_ );
      }
      else if ( arg == "--burnin" ) {
	process_numeric_argument( argc, argv, i, burnin_ );
	burnin_ = ( burnin_ < 0 ) ? 0 : burnin_;
      }
      else if ( arg == "--eps" ) {
	process_numeric_argument( argc, argv, i, eps_ );
      }
      else if ( arg == "--epsmin" || arg == "--mineps" ) {
	process_numeric_argument( argc, argv, i, epsmin_ );
      }
      else if ( arg == "--euclidean" ) {
	dist_ = gourd::dist_code::euclidean;
      }
      else if ( arg == "--maxit" ) {
	process_numeric_argument( argc, argv, i, maxit_ );
	maxit_ = ( maxit_ <= 0 ) ? 1 : maxit_;
      }
      else if ( arg == "--neighborhood" ) {
	process_numeric_argument( argc, argv, i, nhood_ );
	nhood_ = ( nhood_ <= 0 ) ? -1 : nhood_;
      }
      else if ( arg == "--neighborhood-re" ) {
	process_numeric_argument( argc, argv, i, nhood_rint_ );
	nhood_rint_ = ( nhood_rint_ <= 0 ) ? -1 : nhood_rint_;
      }
      else if ( arg == "--neighborhood-mass" ) {
	process_numeric_argument( argc, argv, i, nhood_mass_ );
	nhood_mass_ = ( nhood_mass_ <= 0 ) ? -1 : nhood_mass_;
      }
      else if ( arg == "--rational-quadratic" || arg == "-rq" ) {
	covar_ = gourd::cov_code::rq;
      }
      else if ( arg == "--matern" ) {
	covar_ = gourd::cov_code::matern;
      }
      else if ( arg == "--targetmh" ) {
	process_numeric_argument( argc, argv, i, target_mh_ );
      }
      else if ( arg == "--nsamples" ) {
	process_numeric_argument( argc, argv, i, nsamples_ );
	nsamples_ = ( nsamples_ < 0 ) ? 0 : nsamples_;
      }
      else if ( arg == "--output" || arg == "-o" ) {
	process_string_argument(
          argc, argv, i, output_basename_ );
      }
      else if ( arg == "--profile" ) {
	profile_ = true;
      }
      else if ( arg == "--radial-basis" || arg == "-rbf" ) {
	covar_ = gourd::cov_code::rbf;
      }
      else if ( arg == "--steps" || arg == "--integrator-steps" ) {
	process_numeric_argument( argc, argv, i, steps_ );
      }
      else if ( arg == "--seed" ) {
	process_numeric_argument( argc, argv, i, seed_ );
      }
      else if ( arg == "--subset" ) {
	process_file_argument( argc, argv, i, subset_file_ );
      }
      else if ( arg == "--surface" || arg == "-surf" ) {
	process_file_argument( argc, argv, i, surface_file_ );
      }
      else if ( arg == "--theta" ) {
	process_vector_argument( argc, argv, i, theta_ );
      }
      else if ( arg == "--thin" ) {
	process_numeric_argument( argc, argv, i, thin_ );
	thin_ = ( thin_ <= 0 ) ? 1 : thin_;
      }
      else if ( arg == "--threads" ) {
	process_numeric_argument( argc, argv, i, threads_ );
	threads_ = ( threads_ <= 0 ) ? 0 : threads_;
      }
      else if ( arg == "--varcomp" ||
		arg == "--variance-components" ) {
	process_string_argument( argc, argv, i, varcomp_indices_ );
      }
      else if ( arg == "--xtol" ) {
	process_numeric_argument( argc, argv, i, xtol_ );
	xtol_ = ( xtol_ <= 1e-8 ) ? 1e-8 : xtol_;
      }
      else if ( is_file(arg) ) {
	metric_files_.push_back( arg );
      }
      else if ( warn_unrecognized && arg.substr(0, 1) == "-" ) {
	std::cerr << "Unrecognized option '" << arg << "'\n";
      }
      else {
	std::cerr << "Unknown argument '" << arg << "'\n";
      }

      // --- end parse options
      
      if ( error() || help_invoked() ) {
	break;
      }
    }
    // for ( int i = 1; i < argc; i++ )
  }
  else {
    this->status_ = call_status::error;
  }
  // if ( argc >= 2 )

  
  /* Make sure required parameters are filled */
  //
  if ( nhood_mass_ <= 0 ) {
    nhood_mass_ = nhood_ / 2;
  }
  
  if ( help_invoked() ) {
    show_help();
  }
  else {
    /* Parse particular error types */

    subset_metric_files_();
    if ( metric_files_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input metric images(s) (*.nii)\n\n";
      this->status_ = call_status::error;      
    }
    if ( surface_file_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input surface (*.gii)\n\n";
      this->status_ = call_status::error;
    }
    if ( eps_ <= 0 ) {
      std::cerr << "\n*** ERROR: "
		<< " HMC initial step size must be > 0 ("
		<< eps_ << " given)\n\n";
      this->status_ = call_status::error;
    }
    if ( nhood_ <= 0 ) {
      std::cerr << "\n*** ERROR: "
		<< " NNGP neighborhood must be positive ("
		<< nhood_ << " given)\n\n";
      this->status_ = call_status::error;
    }
    if ( nhood_rint_ <= 0 ) {
      std::cerr << "\n*** ERROR: "
		<< " NNGP neighborhood must be positive ("
		<< nhood_rint_ << " given)\n\n";
      this->status_ = call_status::error;
    }
    if ( nhood_mass_ <= 0 ) {
      std::cerr << "\n*** ERROR: "
		<< " NNGP neighborhood must be positive ("
		<< nhood_mass_ << " given; mass)\n\n";
      this->status_ = call_status::error;
    }
    if ( steps_ <= 0 ) {
      std::cerr << "\n*** ERROR: "
		<< " Integrator steps must be positive ("
		<< steps_ << " given)\n\n";
      this->status_ = call_status::error;
    }
    if ( target_mh_ <= 0 || target_mh_ >= 1 ) {
      std::cerr << "\n*** ERROR: "
		<< " Metropolis-Hastings target rate must be on "
		<< "(0, 1); (" << target_mh_ << " given)\n\n";
      this->status_ = call_status::error;      
    }
    if ( theta_.empty() ) {
      theta_.resize(3);
      switch (covar_) {
        case gourd::cov_code::rbf    : {
	  theta_[0] = 1; theta_[1] = 0.231; theta_[2] = 1; break;
	}
        case gourd::cov_code::rq     : {
	  theta_[0] = 1; theta_[1] = 16; theta_[2] = 1; break;
	}
        case gourd::cov_code::matern : {
	  theta_[0] = 1; theta_[1] = 4.328; theta_[2] = 0.5; break;
	}
      };
    }
  }
  // if ( help_invoked() )

  
  /* Additional error checking */
  if ( target_mh_ <= 0 || target_mh_ > 1 ) {
    std::cerr << "\n*** Warning: --targetmh should be between (0, 1]."
	      << " Reverting to default\n\n";
    target_mh_ = 0.65;
  }
  

  if ( error() ) {
    show_usage();
    std::cerr << "See " << caller() << " --help for more information\n";
  }
};
// glm_command_parser( const int argc, const char *argv[] )









bool gourd::glm_command_parser::profile_computation() const {
  return profile_;
};

double gourd::glm_command_parser::eps() const {
  return eps_;
};

double gourd::glm_command_parser::eps_min() const {
  return epsmin_;
};

double gourd::glm_command_parser::metropolis_target() const {
  return target_mh_;
};

double gourd::glm_command_parser::neighborhood() const {
  return nhood_;
};

double gourd::glm_command_parser::neighborhood_random_intercept()
  const {
  return nhood_rint_;
};

double gourd::glm_command_parser::neighborhood_mass() const {
  return nhood_mass_;
};

double gourd::glm_command_parser::optim_xtol() const {
  return xtol_;
};

gourd::cov_code gourd::glm_command_parser::cov_function() const {
  return covar_;
};

gourd::dist_code gourd::glm_command_parser::distance_metric() const {
  return dist_;
};

int gourd::glm_command_parser::integrator_steps() const {
  return steps_;
};

int gourd::glm_command_parser::mcmc_burnin() const {
  return burnin_;
};

int gourd::glm_command_parser::mcmc_nsamples() const {
  return nsamples_;
};

int gourd::glm_command_parser::mcmc_thin() const {
  return thin_;
};

int gourd::glm_command_parser::optim_maxit() const {
  return maxit_;
};

int gourd::glm_command_parser::threads() const {
  return threads_;
};


unsigned gourd::glm_command_parser::seed() const {
  return seed_;
};

std::string gourd::glm_command_parser::covariate_file() const {
  return covariate_file_;
};

std::string gourd::glm_command_parser::output_basename() const {
  return output_basename_;
};

std::string gourd::glm_command_parser::subset_file() const {
  return subset_file_;
};

std::string gourd::glm_command_parser::surface_file() const {
  return surface_file_;
};

const std::vector<double>& gourd::glm_command_parser::theta() const {
  return theta_;
};

const std::vector<std::string>&
gourd::glm_command_parser::metric_files() const {
  return metric_files_;
};


gourd::ragged_array<int>
gourd::glm_command_parser::variance_component_indices() const {
  return gourd::parse_indices_text(varcomp_indices_);
};



void gourd::glm_command_parser::subset_metric_files_() {
  if ( !subset_file_.empty() && !metric_files_.empty() ) {

    gourd::ragged_array<std::string> tokens =
      abseil::csv_reader<std::string>::read_file( subset_file_ );

    /* Reserve subset to same size as metric_files_ to cover
     * cases where, e.g., multiple files match one token */
    std::vector<std::string> subset;
    subset.reserve( metric_files_.size() );
    int matched_tokens = 0;
    for ( auto& row : tokens ) {
      const std::string tok = row[0];
      int nmatches = 0;
      for ( const std::string& file : metric_files_ ) {
	if ( file.find(tok) != std::string::npos ) {
	  if ( subset.empty() ) {
	    subset.push_back(file); nmatches++;
	  }
	  else {  // Only push_back file if not already in subset
	    int k = 0; bool duplicated = false;
	    while (k < matched_tokens && !duplicated) {
	      duplicated = subset[k] == file;
	      k++;
	    }
	    if ( !duplicated ) {
	      subset.push_back(file); nmatches++;
	    }
	  }
	}  // if ( file.find(tok) != std::string::npos )
      }  // for ( const std::string& file ...
      if (nmatches) matched_tokens++;
    }  // for ( auto& row : tokens)
    
    subset.shrink_to_fit();
    metric_files_.assign(subset.begin(), subset.end());
    if ( matched_tokens < (int)tokens.size() ) {
      std::cerr << "\n\t*** " << matched_tokens << " out of "
		<< tokens.size() << " tokens matched from file "
		<< subset_file_ << "\n";
    }
  }  // if ( !subset_file.empty() ...
};


#endif  // _GOURD_GLM_COMMAND_PARSER_
