
#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>


#ifndef _ABSEIL_STRING_
#define _ABSEIL_STRING_


namespace abseil {

  inline std::string ltrim(
    const std::string& s,
    const std::string rm = " \n\r\t\f\v"
  ) {
    size_t pos = s.find_first_not_of(rm);
    return ( pos == std::string::npos ) ? "" : s.substr(pos);
  };

  inline std::string rtrim(
    const std::string& s,
    const std::string rm = " \n\r\t\f\v"
  ) {
    size_t pos = s.find_last_not_of(rm);
    return ( pos == std::string::npos ) ? "" : s.substr(0, pos + 1);
  };


  /* ****************************************************************/
  /*! Remove unwanted characters from the front/back of strings
   *
   * @param s   input string
   * @param rm  unwanted characters to remove
   * @return    trimmed string
   */
  inline std::string trim(
    const std::string& s,
    const std::string rm = " \n\r\t\f\v"
  ) {
    return rtrim( ltrim(s, rm), rm );
  };


  
  /* ****************************************************************/
  /*! Search a string for a substring, ignoring case
   *
   * @param str    String to search
   * @param token  String token to match
   * @return  Starting position of \c token in \c str if a match
   *   is found, and \c std::string::npos otherwise
   */
  inline size_t find_ignore_case(
    const std::string& str,
    const std::string& token
  ) {
    const std::string::const_iterator it = std::search(
      str.cbegin(), str.cend(), token.cbegin(), token.cend(),
      []( char ca, char cb ){
	return std::toupper(ca) == std::toupper(cb);
      }
    );
    return ( it == str.cend() ) ?
      std::string::npos : std::distance( str.cbegin(), it );
  };



  /*! Split string on delimeter character
   */
  inline std::vector<std::string> split(
    const std::string& str,
    const char delim
  ) {
    std::stringstream ss( str );
    std::vector<std::string> v;
    std::string atom;
    while ( std::getline(ss, atom, delim) ) {
      v.push_back( atom );
    }
    return v;
  };
  
  
};
// namespace abseil





#endif  // _ABSEIL_STRING_
