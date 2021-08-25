
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "abseil/quantile.hpp"

#include "gourd/utilities/read_log_file.hpp"


#ifndef _GOURD_CREDIBLE_BAND_
#define _GOURD_CREDIBLE_BAND_

namespace gourd {

  template< typename T = float >
  struct band {
    std::vector<T> upper;  /*!< Upper interval boundary */
    std::vector<T> lower;  /*!< Lower interval boundary - should be
			    *    same size as \c upper */
    size_t size() const {
      return (upper.size() < lower.size()) ?
	upper.size() : lower.size();
    };
  };



  /* Currently unused */
  template< int p = Eigen::Infinity >
  std::vector< gourd::band<float> > get_file_credbands(
    const std::string fname,
    const std::vector<float>& prob
  );



  /*! Get simultaneous posterior credible bands from simulation draws
   *
   * Method estimates the \c prob quantiles of the lp-norm of each
   * sample and constructs envelopes around samples with norms <=
   * to these quantiles
   *
   */
  template< int p = Eigen::Infinity >
  std::vector< gourd::band<float> > get_file_credbands_unsc(
    const std::string fname,
    const std::vector<float>& prob,
    const float xmax = 1e8
  );


  template< int p = Eigen::Infinity >
  std::vector<float> get_file_lpnorms(
    const std::string fname
  );

  template< int p = Eigen::Infinity >
  std::vector<float> get_file_lpnorms(
    const std::string fname,
    const std::vector<float>& location,
    const std::vector<float>& scale,
    const float eps = 1e-6
  );



  /*
   *
   * @return \c gourd::band<float> object with the mean in field
   *   \c lower, and and standard error in field \c upper
   */
  gourd::band<float> get_file_mean_stderr( const std::string fname );

  
  template< int p = Eigen::Infinity >
  float get_line_lpnorm(
    const std::string& line,
    const char delim
  );

  
  template< int p = Eigen::Infinity >
  float get_line_lpnorm(
    const std::string& line,
    const std::vector<float>& location,
    const std::vector<float>& scale,
    const char delim,
    const float eps = 1e-6
  );
  
}  // namespace gourd








template< int p >
std::vector< gourd::band<float> > gourd::get_file_credbands(
  const std::string fname,
  const std::vector<float>& prob
) {
  /* Method from Ruppert, Wand, and Carroll, 2003 */
  const gourd::band<float> sstat = gourd::get_file_mean_stderr( fname );
  const std::vector<float> norms =
    gourd::get_file_lpnorms<p>( fname, sstat.lower, sstat.upper );
  const std::vector<float> q = abseil::quantile(
    norms.cbegin(), norms.cend(), prob.cbegin(), prob.cend() );
  std::vector< gourd::band<float> > cbands(q.size());
  const size_t n = sstat.size();
  for ( size_t i = 0; i < q.size(); i++ ) {
    cbands[i].lower.resize(n); cbands[i].upper.resize(n);
    for ( size_t j = 0; j < n; j++ ) {
      float w = q[i] * sstat.upper[j];
      cbands[i].lower[j] = -w + sstat.lower[j];
      cbands[i].upper[j] =  w + sstat.lower[j];
    }
  }
  return cbands;
};






template< int p >
std::vector< gourd::band<float> > gourd::get_file_credbands_unsc(
  const std::string fname,
  const std::vector<float>& prob,
  const float xmax
) {
  const std::vector<float> norms =
    gourd::get_file_lpnorms<p>( fname );
  const std::vector<float> q = abseil::quantile(
    norms.cbegin(), norms.cend(), prob.cbegin(), prob.cend() );
  std::ifstream ifs( fname, std::ifstream::in );
  std::vector< gourd::band<float> > cbands(q.size());
  int lineno = 0;
  if ( ifs.is_open() ) {
    std::string line;
    char delim;
    while (std::getline(ifs, line)) {
      if (lineno == 0) {
	delim = gourd::utilities::detect_delimeter(line);
      }
      std::vector<float> x =
	gourd::utilities::parse_numerical_line( line, delim );
      if (lineno == 0) {
	for ( size_t i = 0; i < prob.size(); i++ ) {
	  cbands[i].upper = std::vector<float>(x.size(), -xmax);
	  cbands[i].lower = std::vector<float>(x.size(),  xmax);
	}
      }
      else if (x.size() != cbands[0].upper.size()) {
	throw std::runtime_error(
          "get_file_credbands: file contains imbalanced rows");	
      }
      for ( size_t i = 0; i < q.size(); i++ ) {
	if (norms[ lineno ] <= q[i]) {
	  for ( size_t j = 0; j < x.size(); j++ ) {
	    cbands[i].upper[j] = std::max(x[j], cbands[i].upper[j]);
	    cbands[i].lower[j] = std::min(x[j], cbands[i].lower[j]);
	  }
	}
      }
      lineno++;
    }  // while (std::getline(ifs, line))
    ifs.close();
  }  // if ( ifs.is_open() )
  return cbands;
};







template< int p >
std::vector<float> gourd::get_file_lpnorms(
  const std::string fname
) {
  std::ifstream ifs( fname, std::ifstream::in );
  std::vector<float> v;
  int lineno = 0;
  if ( ifs.is_open() ) {
    std::string line;
    char delim;
    while (std::getline(ifs, line)) {
      if (lineno == 0) {
	delim = gourd::utilities::detect_delimeter(line);
      }
      float x = gourd::get_line_lpnorm<p>(line, delim);
      v.push_back( x );
      lineno++;
    }
    ifs.close();
  }
  return v;
};



template< int p >
std::vector<float> gourd::get_file_lpnorms(
  const std::string fname,
  const std::vector<float>& location,
  const std::vector<float>& scale,
  const float eps
) {
  std::ifstream ifs( fname, std::ifstream::in );
  std::vector<float> v;
  int lineno = 0;
  if ( ifs.is_open() ) {
    std::string line;
    char delim;
    while (std::getline(ifs, line)) {
      if (lineno == 0) {
	delim = gourd::utilities::detect_delimeter(line);
      }
      float x =
	gourd::get_line_lpnorm<p>(line, location, scale, delim, eps);
      v.push_back( x );
      lineno++;
    }
    ifs.close();
  }
  return v;
};




gourd::band<float> gourd::get_file_mean_stderr(
  const std::string fname
) {
  std::ifstream ifs( fname, std::ifstream::in );
  std::vector<float> first, second;  /* Moments */
  gourd::band<float> sstat;
  int n = 0;
  if ( ifs.is_open() ) {
    std::string line;
    char delim;
    while (std::getline(ifs, line)) {
      if (n == 0) {
	delim = gourd::utilities::detect_delimeter(line);
      }
      std::vector<float> x =
	gourd::utilities::parse_numerical_line(line, delim);
      if (n == 0) {
	first.resize( x.size(), 0.0f );
	second.resize( x.size(), 0.0f );
      }
      else if (x.size() != first.size()) {
	throw std::runtime_error(
          "get_file_mean_stderr: file contains imbalanced rows");
      }
      for ( size_t i = 0; i < x.size(); i++ ) {
	first[i]  += x[i];
	second[i] += x[i] * x[i];
      }
      n++;
    }
    sstat.lower.resize( first.size() );  /* mean */
    sstat.upper.resize( first.size() );  /* std error */
    for ( size_t i = 0; i < first.size(); i++ ) {
      sstat.lower[i] = first[i] / n;
      sstat.upper[i] = second[i] / n - sstat.lower[i]*sstat.lower[i];
      sstat.upper[i] = std::sqrt( sstat.upper[i] );
    }
    ifs.close();
  }  // if ( ifs.is_open() )
  return sstat;
};



template< int p >
float gourd::get_line_lpnorm(
  const std::string& line,
  const char delim
) {
  using map_t = typename Eigen::Map< Eigen::VectorXf >;
  std::vector<float> vraw =
    gourd::utilities::parse_numerical_line(line, delim);
  map_t v( vraw.data(), vraw.size() );
  return v.lpNorm<p>();
};


template< int p >
float gourd::get_line_lpnorm(
  const std::string& line,
  const std::vector<float>& location,
  const std::vector<float>& scale,
  const char delim,
  const float eps
) {
  assert( eps > 0 && "get_line_lpnorm: eps <= 0" );
  using map_t = typename Eigen::Map< Eigen::VectorXf >;
  std::vector<float> vraw =
    gourd::utilities::parse_numerical_line(line, delim);
  assert( vraw.size() == scale.size() &&
	  "get_line_lpnorm: scale dimension mismatch" );
  for ( size_t i = 0; i < vraw.size(); i++ ) {
    vraw[i] = (vraw[i] - location[i]) / (scale[i] + eps);
  }
  map_t v( vraw.data(), vraw.size() );
  return v.lpNorm<p>();
};




/*! Output stream operator for gourd::band objects 
 * 
 * Uses a tab delimeted format
 */
template< typename T >
std::ostream& operator<<(
  std::ostream& os,
  const gourd::band<T>& b
) {
  const size_t n = b.size();
  for ( size_t i = 0; i < n; i++ )
    os << b.lower[i] << "\t" << b.upper[i] << "\n";
  return os;
};




#endif  // _GOURD_CREDIBLE_BAND_
