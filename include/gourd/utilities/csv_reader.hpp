
#include <Eigen/Core>
#include <string>
#include <vector>

#include "abseil/utilities/csv_reader.hpp"


#ifndef _GOURD_CSV_READER_
#define _GOURD_CSV_READER_


namespace gourd {
namespace utilities {


  /*! Read csv file into matrix
   *
   * This is a memory inefficient implementation, but should be
   * reasonably fast for small to medium sized files. Requires
   * about twice the ammount of memory needed to store the matrix.
   *
   * Delimeters are automatically detected from 
   * {comma, space, tab}; '#' is treated as a comment character and
   * lines beginning with this character will be ignored
   *
   * @param fname  Path to *.csv file
   * @return  \c Eigen::Matrix<T> matrix
   */
  template< typename T >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
  read_csv( const std::string fname ) {
    typedef typename std::vector< std::vector<T> > buffer_type;
    typedef typename Eigen::Matrix<
      T, Eigen::Dynamic, Eigen::Dynamic >
      mat_type;
    const buffer_type b = abseil::csv_reader<T>::read_file(fname);
    mat_type M(b.size(), b[0].size());
    for ( int i = 0; i < (int)b.size(); i++ ) {
      for ( int j = 0; j < (int)b[0].size(); j++ ) {
	M.coeffRef(i, j) = b[i][j];
      }
    }
    return M;
  };
  

};
  // namespace utilities
  
};
// namespace gourd



#endif  // _GOURD_CSV_READER_



