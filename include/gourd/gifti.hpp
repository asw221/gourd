
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gifti_io.h"


#ifndef _GOURD_GIFTI_
#define _GOURD_GIFTI_


namespace gourd {
namespace gifti {


  /*! Read gifti image data
   *
   * Read data and convert to float if possible. Declared in file
   * gourd/gifti.hpp
   *
   * @param fname  Path to image
   * @param read   Flag to indicate whether or not to read data
   *               (vs just the header information)
   * @return  Pointer to a \c gifti_image data structure. Will need
   *   to be freed with \c ::gifti_free_image()
   */
  ::gifti_image* image_read(
    const std::string fname,
    const int read = 1
  ) {
    const std::filesystem::path initial_dir =
      std::filesystem::current_path();
    const std::filesystem::path fp( fname );
    ::gifti_image* gim = NULL;
    try {
      std::filesystem::current_path( fp.parent_path() );
      gim = ::gifti_read_image( fp.filename().c_str(), read );
      ::gifti_convert_to_float( gim );
    }
    catch ( const std::exception& ex ) {
      std::cerr << "\t*** Error reading "
		<< fname
		<< ":\n\t"
		<< ex.what()
		<< std::endl;
    }
    std::filesystem::current_path( initial_dir );
    return gim;
  };

};
  // namespace gifti

};
// namespace gourd


#endif  // _GOURD_GIFTI_
