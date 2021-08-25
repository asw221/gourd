
#include <sstream>
#include <string>
#include <vector>

#include "abseil/utilities/csv_reader.hpp"


#ifndef _GOURD_READ_LOG_FILE_
#define _GOURD_READ_LOG_FILE_

namespace gourd {
namespace utilities {


  /*! Detect delimeting \c char in a line of text
   */
  inline char detect_delimeter( const std::string& line ) {
    return abseil::csv_reader<float>::detect_delimeter(line);
  };
  

  /*! Parse numerical data with gourd log-file format
   *
   * (Does not error check numerical conversions)
   * (Force data into single precision format)
   */
  std::vector<float> parse_numerical_line(
    const std::string& line,
    const char delim
  ) {
    std::stringstream ss( line );
    std::vector<float> v;
    std::string atom;
    while ( std::getline(ss, atom, delim) )
      v.push_back( std::stof(atom) );
    return v;
  };
  


  
}  // namespace utilities
}  // namespace gourd

#endif  // _GOURD_READ_LOG_FILE_
