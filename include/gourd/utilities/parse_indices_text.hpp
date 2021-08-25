
#include <string>
#include <vector>

#include "abseil/utilities/string.hpp"

#include "gourd/utilities/ragged_array.hpp"


#ifndef _GOURD_PARSE_INDICES_TEXT_
#define _GOURD_PARSE_INDICES_TEXT_

namespace gourd {

  /*! Parse specially formatted list of indices
   *
   * Treats colons (:) as delimiting blocks of indices, and 
   * commas (,) as delimiting indices within block. Also interprets
   * ranges of indices given in X-Y format. For example,
   * \c gourd::parse_indices_text("0,1,2:3-5,8:6-7")
   * is interpreted into the ragged \c int array,
   * \c {{0,1,2},{3,4,5,8},{6,7}}
   *
   * @return A \c gourd::ragged_array<int> object containing integer
   *   indices grouped by row
   */
  gourd::ragged_array<int> parse_indices_text( const std::string& s );


  /* Interpret integer text delimited by commas */
  void parse_indices_text_block(
    const std::string& s,
    std::vector<int>& out  /* Modified */
  );

  /* Interpret integer ranges: "0-2" ==> {0,1,2} */
  void expand_indices_text_dash(
    const std::string& s,
    std::vector<int>& out  /* Modified */
  );
  
}  // namespace gourd




gourd::ragged_array<int> gourd::parse_indices_text(
  const std::string& s
) {
  std::vector<std::string> initial_split = abseil::split(s, ':');
  ragged_array<int> result(initial_split.size());
  for (size_t i = 0; i < initial_split.size(); i++)
    gourd::parse_indices_text_block(initial_split[i], result[i]);
  return result;
};


void gourd::parse_indices_text_block(
  const std::string& s,
  std::vector<int>& out
) {
  for (std::string& sub : abseil::split(s, ',')) {
    try { gourd::expand_indices_text_dash(sub, out); }
    catch (...) { ; }
  }
};

void gourd::expand_indices_text_dash(
  const std::string& s,
  std::vector<int>& out
) {
  const size_t p = s.find('-');
  if ( p != std::string::npos ) {
    int first = std::stoi( s.substr(0u, p) );
    int last = std::stoi( s.substr(p+1, s.size()-1) );
    if (last < first) {
      int temp = first; first = last; last = temp;
    }
    for (int i = first; i <= last; i++)  out.push_back(i);
  }
  else {
    out.push_back(std::stoi(s));
  }
};

#endif  // _GOURD_PARSE_INDICES_TEXT_
