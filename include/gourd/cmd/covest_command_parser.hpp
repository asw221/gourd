
#include <iostream>
#include <string>
#include <vector>

#include "gourd/options.hpp"
#include "gourd/cmd/basic_command_parser.hpp"


#ifndef _GOURD_COVEST_COMMAND_PARSER_
#define _GOURD_COVEST_COMMAND_PARSER_

namespace gourd {

  class covest_command_parser :
    public gourd::basic_command_parser {
  public:
    covest_command_parser( const int argc, const char** argv );

    void show_help() const override;
    void show_usage() const override;

    bool rescale() const;

    double neighborhood() const;
    double tol() const;

    int maxit() const;
    int print_level() const;
    int which_params() const;

    gourd::dist_code distance_metric() const;
    gourd::cov_code  cov_function() const;
    
    std::string surface_file() const;
    const std::vector<std::string>& metric_files() const;
    const std::vector<double>& theta() const;

  private:
    bool   rescale_;
    
    double rad_;
    double tol_;
    
    int    maxit_;
    int    which_params_;
    int    print_level_;
    
    gourd::dist_code dist_;
    gourd::cov_code  cov_;
    
    std::string surface_file_;
    std::vector<std::string> metric_files_;
    
    std::vector<double> theta_;

    void set_theta_default();
    
  };

}  // namespace gourd



void gourd::covest_command_parser::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "  --surface      file/path  REQURED. GIFTI surface file \n"
	    << "  --radius       float      Nearest-Neighbor Gaussian Process radius \n"
	    << "  --rescale                 Rescale outcome images? (Default = false) \n"
	    << "  --maxit        int        (Default = 5000) \n"
	    << "  --print-level  int        0: Silent -- 2: Verbose (Default) \n"
	    << "  --start        float[3]   Optimization starting point \n"
	    << "  --tol          float      Optimization tolerance (Default = 1e-6) \n"
	    << "  --which        int        1: Variance; 2: +Width; 3: +Smoothness \n"
	    << "\n"
	    << "Covariance Functions:\n"
	    << "  --radial-basis       (Default) \n"
	    << "  --rational-quadratic \n"
	    << "  --matern \n"
	    << "\n"
	    << "Distance Metrics:\n"
	    << "  --great-circle       (Default) \n"
	    << "  --euclidean \n"
	    << "\n"
	    << "\nUNUSED\n"
	    << "  --covariates  file/path  UNUSED. Mean model covariates \n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << std::endl;
};


void gourd::covest_command_parser::show_usage() const {
  std::cerr << "Usage:\n"
	    << "\t" << this->caller()
	    << "  path/to/data*.nii --surface path/to/surf.gii <options> \n"
	    << std::endl;
};



gourd::covest_command_parser::covest_command_parser(
  const int argc, const char * argv[]
) : gourd::basic_command_parser(argc, argv)
{
  /* Default values */
  // -----------------------------------------------------------------
  rad_   = 10;
  tol_   = 1e-6;
  maxit_ = 500;
  print_level_  = 2;
  which_params_ = 3;
  rescale_ = false;
  dist_  = gourd::dist_code::great_circle;
  cov_   = gourd::cov_code::rbf;
  // -----------------------------------------------------------------

  /* Parse inputs */
  if ( argc >= 2 ) {
    for ( int i = 1; i < argc; i++ ) {
      //
      std::string arg = argv[i];
      //
      if ( arg == "-h" || arg == "--help" ) {
	this->status_ = call_status::help;
	break;
      }
      else if ( arg == "--rescale" ) {
	rescale_ = true;
      }
      else if ( arg == "--euclidean" ) {
	dist_ = gourd::dist_code::euclidean;
      }
      else if ( arg == "--great-circle" || arg == "--geodesic" ) {
	dist_ = gourd::dist_code::great_circle;
      }
      else if ( arg == "--radial-basis" || arg == "-rbf" ) {
	cov_ = gourd::cov_code::rbf;
      }
      else if ( arg == "--radius" || arg == "--neighborhood" ) {
	this->process_numeric_argument( argc, argv, i, rad_ );
      }
      else if ( arg == "--rational-quadratic" || arg == "-rq" ) {
	cov_ = gourd::cov_code::rq;
      }
      else if ( arg == "--matern" ) {
	cov_ = gourd::cov_code::matern;
      }
      else if ( arg == "--maxit" ) {
	this->process_numeric_argument( argc, argv, i, maxit_ );
      }
      else if ( arg == "--print-level" || arg == "--verbosity" ) {
	this->process_numeric_argument( argc, argv, i, print_level_ );
	print_level_ = (print_level_ > 3 || print_level_ < 0) ?
	  0 : print_level_;
      }
      else if ( arg == "--surface" ) {
	this->process_file_argument( argc, argv, i, surface_file_ );
      }
      else if ( arg == "--start" || arg == "--theta" ) {
	this->process_vector_argument( argc, argv, i, theta_ );
	if ( theta_.size() != 3 ) {
	  std::cerr << arg << " must be followed by 3 numeric "
		    << "arguments\n";
	  this->status_ = call_status::error;
	}
      }
      else if ( arg == "--tol" ) {
	this->process_numeric_argument( argc, argv, i, tol_ );
	tol_ = ( tol_ < 0 ) ? -tol_ : tol_;
      }
      else if ( arg == "--which" || arg == "-w" ) {
	this->process_numeric_argument( argc, argv, i, which_params_);
	which_params_ = (which_params_ <= 0 || which_params_ > 3) ?
	  3 : which_params_;
      }
      else if ( this->is_file(arg) ) {
	metric_files_.push_back( arg );
      }
      else if ( arg.substr(0, 1) == "-" ) {
	std::cerr << "Unrecognized option '" << arg << "'\n";
      }
      else {
	std::cerr << "Unknown argument '" << arg << "'\n";
      }

      if ( this->error() || this->help_invoked() ) {
	break;
      }
    }
    // for ( int i = 1; i < argc; i++ )
  }
  else {
    this->status_ = call_status::error;
  }
  // if ( argc >= 2 ) / else

  if ( this->help_invoked() ) {
    show_help();
  }
  else {
    if ( metric_files_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input metric image(s) (*.nii)\n\n";
      this->status_ = call_status::error;
    }
    if ( surface_file_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input surface (*.gii)\n\n";
      this->status_ = call_status::error;
    }
    //
    if ( theta_.empty() ) {
      set_theta_default();
    }
  }
  // if ( this->help_invoked() ) / else

  if ( this->error() ) {
    show_usage();
    std::cerr << "See " << this->caller()
	      << " --help for more information\n";
  }
};




void gourd::covest_command_parser::set_theta_default() {
  theta_.resize(3);
  switch (cov_) {
    case gourd::cov_code::rbf    : {
      theta_[0] = 1; theta_[1] = 0.231; theta_[2] = 1;
      break;      
    }
    case gourd::cov_code::rq     : {
      theta_[0] = 1; theta_[1] = 16; theta_[2] = 1;
      break;
    }
    case gourd::cov_code::matern : {
      theta_[0] = 1; theta_[1] = 4.328; theta_[2] = 0.5;
    }
  };
};


bool gourd::covest_command_parser::rescale() const {
  return rescale_;
};


double gourd::covest_command_parser::tol() const {
  return tol_;
};

double gourd::covest_command_parser::neighborhood() const {
  return rad_;
};

int gourd::covest_command_parser::maxit() const {
  return maxit_;
};

int gourd::covest_command_parser::print_level() const {
  return print_level_;
};

int gourd::covest_command_parser::which_params() const {
  return which_params_;
};


gourd::dist_code gourd::covest_command_parser::distance_metric()
  const {
  return dist_;
};

gourd::cov_code gourd::covest_command_parser::cov_function()
  const {
  return cov_;
};



std::string gourd::covest_command_parser::surface_file() const {
  return surface_file_;
};

const std::vector<std::string>&
gourd::covest_command_parser::metric_files() const {
  return metric_files_;
};

const std::vector<double>&
gourd::covest_command_parser::theta() const {
  return theta_;
};

#endif  // _GOURD_COVEST_COMMAND_PARSER_

