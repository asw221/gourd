
#include <iostream>

#include "nifti2_io.h"
#include "afni_xml_io.h"

#include "gourd/cifti_xml.hpp"



int main ( int argc, char * argv[] ) {

  int out = 0;
  
  if ( argc < 2 ) {
    std::cerr << "\nUsage:\n\tcifti_info /path/to/img\n";
    out = -1;
  }

  ::nifti_image* nim = ::nifti_image_read( argv[1], 0 );
  
  if ( !::nifti_looks_like_cifti( nim ) ) {
    std::cerr << "Input image " << argv[1]
	      << " does not have CIFTI specs\n";
    out = -1;
  }
  else {

    ::afni_xml_t* ext = ::axio_cifti_from_ext( nim );
    gourd::display_cifti_xml( ext );
    ::axml_free_xml_t( ext );
    
  }

  ::nifti_image_free( nim );
  return out;
  
}


