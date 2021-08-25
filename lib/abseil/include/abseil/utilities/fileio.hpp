
#include <cstdio>
#include <string>


#ifndef _ABSEIL_FILEIO_
#define _ABSEIL_FILEIO_

namespace abseil {

  /*! Return number of lines in a basic file 
   * 
   * Implementation counts the number of newline characters, '\n'
   */
  long lines_in_file( const std::string fname ) {
    FILE* ifile = fopen( fname.c_str(), "r" );
    if ( ifile ) {
      long num_lines = 0;
      int ch;
      while ( EOF != (ch = getc(ifile)) ) {
	if ( ch == '\n' ) num_lines++;
      }
      fclose( ifile );
      return num_lines;
    }
    return -1;
  };

}  // namespace abseil


#endif  // _ABSEIL_FILEIO_

