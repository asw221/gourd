
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "abseil/utilities/string.hpp"


#ifndef _ABSEIL_CSV_READER_
#define _ABSEIL_CSV_READER_


namespace abseil {

  /* ****************************************************************/
  /*! Read single type data from delimited files
   *
   * Delimeting character automatically detected from first
   * non-comment line, or will default to ',' if none is found.
   *
   * Comment characters are searched by exact match.
   *
   */
  template< typename T >
  class csv_reader {    
  public:

    /*! Process ordinary file with single type delimeted data
     *
     * Default value of \c quote ignores double and single quotation
     * marks as well as unicode right and left double and single
     * quotation marks.
     *
     * @return  an \c std::vector<std::vector<T>> object containing
     *   the processed data. Outer indices are over rows; inner
     *   indices columns.
     */
    static std::vector< std::vector<T> > read_file(
      const std::string filename,
      const std::string delim_opts = " ,\t",
      const std::string comment = "#",
      const std::string quote = "\"\'\u201C\u201D\u2018\u2019"
    );

    
    /*! Finds and returns first matching instance in \c delim_opts */
    static char detect_delimeter(
      const std::string& line,
      const std::string delim_opts = " ,\t"
    );  
  
  protected:

    /* Parse line input
     * Separate by \c delimeter and remove leading/trailing \c quote 
     * characters. Parsed out values are appended to \c data
     */
    static bool parse_line_(
      std::istringstream& line,
      std::vector<T>& data,
      const char delimeter,
      const std::string quote
    );

    /* Checks first n characters for \c comment sequence */
    static bool is_comment_line_(
      const std::string& line,
      const std::string& comment
    );

  
  };
  // class csv_reader
  /* ****************************************************************/


};
// namespace abseil





template<>
bool abseil::csv_reader<std::string>::parse_line_(
  std::istringstream& line,
  std::vector<std::string>& data,
  const char delimeter,
  const std::string quote
) {
  while ( line ) {
    std::string atom;
    if ( std::getline(line, atom, delimeter) ) {
      data.push_back( abseil::trim(atom, quote) );
    }
  }
  return !data.empty();
};


template< typename T >
bool abseil::csv_reader<T>::parse_line_(
  std::istringstream& line,
  std::vector<T>& data,
  const char delimeter,
  const std::string quote
) {
  while ( line ) {
    std::string atom;
    if ( std::getline(line, atom, delimeter) ) {
      try {
	data.push_back( (T)std::stod(abseil::trim(atom, quote)) );
      }
      catch (...) { ; }  // <- add to later?
    }
  }
  return !data.empty();
};



template< typename T >
bool abseil::csv_reader<T>::is_comment_line_(
  const std::string& line,
  const std::string& comment
) {
  return line.substr(0, comment.size()) == comment;
};



template< typename T >
char abseil::csv_reader<T>::detect_delimeter(
  const std::string& line,
  const std::string delim_opts
) {
  size_t i = line.find_first_of(delim_opts);
  return ( i == std::string::npos ) ? ',' : line[i];
};




template< typename T >
std::vector< std::vector<T> >
abseil::csv_reader<T>::read_file(
  const std::string filename,
  const std::string delim_opts,
  const std::string comment,
  const std::string quote
) {
  std::ifstream ifile( filename.c_str(), std::ifstream::in );
  std::vector< std::vector<T> > data;
  int lineno = 0;
  char delim;
  if ( ifile ) {
    while ( ifile ) {
      std::string line;
      if ( std::getline(ifile, line) ) {
	if ( !is_comment_line_(line, comment) ) {
	  if ( lineno == 0 ) {
	    delim = detect_delimeter( line, delim_opts );
	  }
	  std::istringstream liness(line);
	  std::vector<T> line_data;
	  if ( parse_line_(liness, line_data, delim, quote) ) {
	    data.push_back( line_data );
	  }
	  lineno++; 
	}
      }
    }  // while ( ifile )
  }
  else {
    std::string msg = "Could not open file: ";
    throw std::runtime_error( msg + filename );
  }
  ifile.close();
  return data;
};



#endif  // _ABSEIL_CSV_READER_


