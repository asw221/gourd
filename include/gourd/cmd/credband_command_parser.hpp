
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "gourd/cmd/basic_command_parser.hpp"


#ifndef _GOURD_CREDBAND_COMMAND_PARSER_
#define _GOURD_CREDBAND_COMMAND_PARSER_

namespace gourd {

  /* ****************************************************************/
  /*! Parse command line input for \c gourd_credband 
   */
  class credband_command_parser :
    public gourd::basic_command_parser {
  public:
    using call_status = gourd::basic_command_parser::call_status;
    using gourd::basic_command_parser::caller;
    using gourd::basic_command_parser::error;
    using gourd::basic_command_parser::help_invoked;

    credband_command_parser( const int argc, const char** argv );
    
    void show_help() const override;
    void show_usage() const override;

    std::string logfile() const;
    std::string output_name(
      const float width,
      const float scale = 100
    ) const;
    std::string reference_image() const;
    
    const std::vector<float>& p() const;

  private:
    std::string logfile_;
    std::string ref_img_;
    std::vector<float> p_;
  };
  

}  // namespace gourd



/* Help pages */
void gourd::credband_command_parser::show_help() const {
  show_usage();
  std::cerr << "Options\n"
	    << "  --reference  file/path  REQUIRED. CIFTI format reference image \n"
	    << "  --width      p1 p2 ...  Credible band widths on [0, 1]\n"
	    << "\n"
	    << "  -w   | -p    p1 p2 ...  Aliases for --width\n"
	    << "  -ref | -r    file/path  Aliases for --reference\n"
	    << "\n"
	    << std::endl;
};


void gourd::credband_command_parser::show_usage() const {
  std::cerr << "Usage:\n"
	    << "\t" << caller()
	    << " path/to/logfile.dat -ref path/to/reference.dtseries.nii"
	    << " -p p1 p2 ...\n"
	    << std::endl;
};



/* Constructor */

gourd::credband_command_parser::credband_command_parser(
  const int argc,
  const char *argv[]
) :
  gourd::basic_command_parser( argc, argv )
{
  if ( argc >= 2 ) {
    for ( int i = 1; i < argc; i++ ) {

      std::string arg = argv[i];

      if ( arg == "-h" || arg == "--help" ) {
	this->status_ = call_status::help;
	break;
      }
      else if ( arg == "--reference" || arg == "-ref"
		|| arg == "-r" ) {
	this->process_file_argument( argc, argv, i, ref_img_ );
      }
      else if ( arg == "--width" || arg == "-p" || arg == "-w" ) {
	this->process_vector_argument( argc, argv, i, p_ );
	for ( float& w : p_ )
	  w = (w <= 1) ? ((w >= 0) ? w : 0.0f) : 1.0f;
      }
      else if ( is_file(arg) ) {
	logfile_ = arg;
      }
      else if ( arg.substr(0, 1) == "-" ) {
	std::cerr << "Unrecognized option '" << arg << "'\n";
      }
      else {
	std::cerr << "Unknown argument '" << arg << "'\n";
      }
      // --- end parse options
      if ( error() || help_invoked() ) {
	break;
      }
    }  // for ( int i = 1; i < argc; i++ )
  }  // if ( argc >= 2 )
  else {
    this->status_ = call_status::error;
  }

  if ( help_invoked() ) {
    show_help();
  }
  else {
    if ( logfile_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply input log file\n\n";
      this->status_ = call_status::error;
    }
    if ( ref_img_.empty() ) {
      std::cerr << "\n*** ERROR: "
		<< " User must supply reference CIFTI file\n\n";
      this->status_ = call_status::error;
    }
    
    /* --- Default values ----------------------------------------- */
    if ( p_.empty() )
      p_ = std::vector<float>(1, 0.95);
    /* ------------------------------------------------------------ */
  }

  if ( error() ) {
    show_usage();
    std::cerr << "See " << caller() << " --help for more information\n";
  }
};
// credband_command_parser( const int argc, const char* argv[] )



/* Getters */

std::string gourd::credband_command_parser::output_name(
  const float width,
  const float scale
) const {
  std::filesystem::path opath(logfile_);
  std::ostringstream fnamess;
  fnamess << opath.stem().string() << "_credband"
	  << std::setprecision(0) << std::fixed
	  << (width * scale);
  opath.replace_filename( fnamess.str() );
  opath.replace_extension( ".dtseries.nii" );
  return opath.string();
};


std::string gourd::credband_command_parser::logfile() const {
  return logfile_;
};

const std::vector<float>& gourd::credband_command_parser::p() const {
  return p_;
};

std::string gourd::credband_command_parser::reference_image() const {
  return ref_img_;
};


#endif  // _GOURD_CREDBAND_COMMAND_PARSER_
