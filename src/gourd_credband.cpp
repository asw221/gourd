
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nifti2_io.h"

#include "ansi.hpp"

#include "gourd/credible_band.hpp"
#include "gourd/nifti2.hpp"
#include "gourd/cmd/credband_command_parser.hpp"


/* Example call:
 * gourd_credbands logfilename.dat ref.dtseries.nii -p p1 p2 ...
 */


int main( const int argc, const char* argv[] ) {

  gourd::credband_command_parser input( argc, argv );
  if ( !input )  return 1;
  else if ( input.help_invoked() )  return 0;

  try {

    ::nifti_image* refnim =
      gourd::nifti2::image_read( input.reference_image(), 0 );

    ::nifti_image* outnim = gourd::nifti2::create_cifti( refnim, 2 );

    std::vector< gourd::band<float> > cbs =
      gourd::get_file_credbands<>( input.logfile(), input.p() );

    if ( refnim->nvox < (int64_t)cbs[0].size() ) {
      throw std::domain_error(
        "Reference image has fewer vertices than log-file" );
    }

    // Deep copy credible bands
    int j = 0;
    float* const data_ptr = static_cast<float*>( outnim->data );
    for ( const gourd::band<float>& band : cbs ) {
      for ( size_t i = 0; i < band.size(); i++ ) {
	int stride = i * 2;
	*(data_ptr + stride) = band.lower[i];
	*(data_ptr + stride + 1) = band.upper[i];
      }
      std::string fname = input.output_name( input.p()[j] );
      gourd::nifti2::image_write( outnim, fname );
      j++;
    }

    // Clean up
    ::nifti_image_free( refnim );
    ::nifti_image_free( outnim ); 
  }
  catch( const std::exception& ex ) {
    std::cerr << ansi::bold << ansi::magenta
	      << "*** Exception caught:\n"
	      << ansi::reset
	      << ex.what()
	      << std::endl;
    return 1;
  }
  catch( ... ) {
    std::cerr << ansi::bold << ansi::magenta
	      << "*** Program error (unknown cause)"
	      << ansi::reset
	      << std::endl;
    return 1;    
  }
};

