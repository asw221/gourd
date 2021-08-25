
#include <cmath>
#include <type_traits>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
// #include <Eigen/SparseCholesky>

#include "abseil/covariance_functors.hpp"
#include "abseil/coordinates.hpp"
#include "abseil/timer.hpp"

#include "gourd/options.hpp"  // gourd::dist_code


#ifndef _GOURD_NEAREST_NEIGHBOR_PROCESS_
#define _GOURD_NEAREST_NEIGHBOR_PROCESS_


namespace gourd {
  

  /* ****************************************************************/
  /*! Nearest neighbor gaussian process Hessian matrix
   *  
   * Stores the LDLT decomposition of Hessian matrix when the NNGP
   * parameter of interest is a random field. Exposes methods for
   * efficient computation of associated matrix products.
   * For example, if
   *                    f(x) ~ N(0, C),
   * with C^-1 sparse (per NNGP), \c nnp_hess stores the sparse
   * LDLT decomposition
   *                   C^-1 = Lt' diag(d) Lt,
   * where Lt is the lower triangular factor of C^-1.
   *
   * Defined in file nearest_neighbor_process.hpp
   */
  template< typename T >  /* General scalar type (real) */
  class nnp_hess {
  public:
    typedef T real_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1>
      vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      mat_type;
    typedef typename Eigen::SparseMatrix<T> spmat_type;


    /*! (constructor) */
    template< size_t D >  /* Dimension of covariance parameter */
    nnp_hess(
      const std::vector< abseil::cartesian_coordinate<3, T> >& coords,
      const abseil::covariance_functor<T, D>* const cov,
      const T r,
      const gourd::dist_code dc = gourd::dist_code::great_circle
    );
    /* TODO: pointers for arbitrary coords containers */

    nnp_hess() = default;

    /*
     * Copy and assignment methods removed for now:
     *   - Eigen has not defined a copy constructor for 
     *     SparseLU<T>, so the decompositions of Lt and Ut
     *     would need to be recomputed (expensive)
     *
    nnp_hess( const nnp_hess& other );    
    nnp_hess<T>& operator=( const nnp_hess<T>& other );
    */
    nnp_hess<T>& operator=( const nnp_hess<T>&& other ) = default;


    /*! Compute log determinant of Hessian.
     *
     * The type S ensures numerical precision. Double precision
     * (the default) should be fine for most applications unless
     * the NNGP Hessian is extremely high dimensional
     */
    template< typename S = double >  /* Accumulator type */
    double ldet() const;

    /*! Quadratic form */
    template< typename S = double >
    double qf( const vector_type& v ) const;

    /*! Quadratic form with inverse Hessian */
    template< typename S = double >
    double iqf( const vector_type& v ) const;

    /*! Trace of quadratic form:  tr(m' Lt' D Lt m) */
    template< typename S = double >
    double trqf( const mat_type& m ) const;

    
    /*! Trace of quadratic form with inverse Hessian:
     * tr(m' Lt^-1 D^-1 Ut^-1 m)
     */
    template< typename S = double >
    double triqf( const mat_type& m ) const;

    /*! Rank of the Hessian */
    int rank() const;

    /*! Half product: Lt' D^1/2 v */
    mat_type hprod( const mat_type& m ) const;
    
    /*! Right multiply Hessian by matrix */
    mat_type rmul( const mat_type& m ) const;

    /*! Inverse half product: Lt^-1 D^-1/2 v */
    mat_type ihprod( const mat_type& m ) const;

    /*! Right multiply inverse Hessian by matrix */
    mat_type irmul( const mat_type& m ) const;

    
    /*! Compute full Hessian matrix 
     * There should not really be a need for this method in most cases
     */
    spmat_type hessian() const;
    
    
  private:

    /* 
     * Compute the vector squared Euclidean norm.
     * Numerically precise if the base scalar_type 
     * is float
     */
    template< typename S = double >  /* Accumulator type */
    T vsqnorm_( const vector_type& v ) const ;

    
    /* Diagonal factor of Hessian matrix */
    vector_type d_;

    /* Lower triangular factor of Hessian matrix */
    spmat_type  Lt_;

    /* Decompositions of lower and upper triangular matrix factors.
     * Used to compute inverse products 
     */
    Eigen::SparseLU<spmat_type> Lt_decomp_;
    Eigen::SparseLU<spmat_type> Ut_decomp_;
  };
  /* ****************************************************************/

  
};
// namespace gourd







template< typename T >  /* Scalar type */
template< size_t D >  /* Dimension of covariance parameter */
gourd::nnp_hess<T>::nnp_hess(
  const std::vector< abseil::cartesian_coordinate<3, T> >& coords,
  const abseil::covariance_functor<T, D>* const cov,
  const T r,
  const gourd::dist_code dc
) {
  static_assert(std::is_floating_point_v<T>,
		"nnp_hess only defined for floating-point types");
  typedef typename Eigen::Triplet<T> triplet_type;
  /*
   * For faster computation with great-circle distances:
   *
   * - Start with coordinates in xyz (easy/native)
   *
   * - Approximate the number of points within a circular neighborhood 
   *    - Area of a small circle on a sphere:
   *      (small circle radius - r; radius of sphere - R)
   *      A = 2 * pi * R^2 * { 1 - cos(r/R) }
   *
   * - Relationship between chord length and GCD:
   *     GCD = R * CA  (Central Angle)
   *      CA = 2 * asin( |chord| / (2 * R) )
   *
   *    - Can compute dx, dy, dz first and test for 
   *      neighborhoodness/weed out
   *
   */

  const T surf_radius = abseil::to_spherical(coords[0]).radius();
  const T maxcl = 2 * surf_radius *
    std::sin( r / (surf_radius * 2) );
  const T nbr_ratio = (1 - std::cos(r / surf_radius)) / 2;
  const T compar = ( dc == gourd::dist_code::great_circle ) ?
    maxcl : r;
  const T c0 = (*cov)(0) ;
    
  int reserve_n = coords.size() * coords.size() * nbr_ratio / 2;
  int prev_nbrs = 1;
  T dx, dy, dz, chord, distance;

  // Initialize output
  std::vector<triplet_type> trips;
  trips.reserve( reserve_n );
  
  d_  = vector_type::Constant( coords.size(), 1 / c0 );
  Lt_ = spmat_type( coords.size(), coords.size() );
  //

  //
  std::cout << "Computing neighborhoods..." << std::flush;
  abseil::timer::start();
  //
  
  trips.push_back( triplet_type(0, 0, (T)1) );
  /* Loop over lower triangle of I - A */
  for ( size_t i = 1; i < coords.size(); i++ ) {
    reserve_n = prev_nbrs * 11 / 10;
    std::vector<T> cs;
    std::vector<int> nbrs;
    cs.reserve( reserve_n );
    nbrs.reserve( reserve_n );
    for ( size_t j = 0; j < i; j++ ) {
      dx = std::abs( coords[i][0] - coords[j][0] );
      dy = std::abs( coords[i][1] - coords[j][1] );
      dz = std::abs( coords[i][2] - coords[j][2] );
      if ( dx <= maxcl &&
	   dy <= maxcl &&
	   dz <= maxcl ) {
	chord = std::sqrt( dx * dx + dy * dy + dz * dz );
	distance = ( dc == gourd::dist_code::great_circle ) ?
	  2 * surf_radius * std::asin( chord / (2 * surf_radius) ) :
	  chord;
	if ( distance <= compar ) {
	  cs.push_back( (*cov)(distance) );
	  nbrs.push_back( j );
	}
      }
    }  // for ( size_t j = 0; j < i; j++ )
    cs.shrink_to_fit();
    nbrs.shrink_to_fit();

    if ( !nbrs.empty() ) {
      vector_type cs_ =
	Eigen::Map<vector_type>( cs.data(), cs.size() );
      mat_type C_( nbrs.size(), nbrs.size() );
      C_.coeffRef(0, 0) = c0;
      for ( int j = 1; j < C_.rows(); j++ ) {
	for ( int k = 0; k < j; k++ ) {
	  chord = coords[ nbrs[j] ].distance(coords[ nbrs[k] ]);
	  distance = ( dc == gourd::dist_code::great_circle ) ?
	    2 * surf_radius * std::asin( chord / (2 * surf_radius) ) :
	    chord;
	  C_.coeffRef(j, k) = (*cov)(distance);
	  C_.coeffRef(k, j) = C_.coeffRef(j, k);
	}
	C_.coeffRef(j, j) = c0;
      }
      vector_type c_tilde = C_.colPivHouseholderQr().solve( cs_ );
      for ( int j = 0; j < c_tilde.size(); j++ ) {
	trips.push_back( triplet_type( i, nbrs[j], -c_tilde.coeffRef(j) ));
      }

      /* Set d_[i] */
      d_.coeffRef(i) = 1 /
	( c0 - (cs_.transpose() * c_tilde)[0] );
	
      prev_nbrs = c_tilde.size();
      prev_nbrs = ( prev_nbrs < 1) ? 1 : prev_nbrs;
    }
    trips.push_back( triplet_type(i, i, (T)1) );
    
  }  // for ( size_t i = 1; i < coords.size(); i++ )

  //
  abseil::timer::stop();
  std::cout << " Done!\n"
	    << "\t(Computation took " << (abseil::timer::diff() / 1e6)
	    << " sec)" << std::endl;
  //

  trips.shrink_to_fit();
  Lt_.setFromTriplets( trips.begin(), trips.end() );

  // -----------------------------------------------------------------

  //
  std::cout << "Computing decomposition of Lt..." << std::flush;
  abseil::timer::start();
  //
  Lt_decomp_.analyzePattern( Lt_ );
  Lt_decomp_.factorize( Lt_ );
  //
  abseil::timer::stop();
  std::cout << " Done!\n"
	    << "\t(Computation took " << (abseil::timer::diff() / 1e6)
	    << " sec)" << std::endl;
  //

  //
  std::cout << "Computing decomposition of Ut..." << std::flush;
  abseil::timer::start();
  //
  Ut_decomp_.analyzePattern( Lt_.adjoint() );
  Ut_decomp_.factorize( Lt_.adjoint() );
  //
  abseil::timer::stop();
  std::cout << " Done!\n"
	    << "\t(Computation took " << (abseil::timer::diff() / 1e6)
	    << " sec)" << std::endl;
  //

};




/*
template< typename T >
gourd::nnp_hess<T>::nnp_hess(
  const gourd::nnp_hess<T>& other
) {
  d_  = other.d_;
  Lt_ = other.Lt_;
  Lt_decomp_ = other.Lt_decomp_;
  Ut_decomp_ = other.Ut_decomp_;
};



template< typename T >
gourd::nnp_hess<T>& gourd::nnp_hess<T>::operator=(
  const gourd::nnp_hess<T>& other
) {
  if ( &other == this ) return *this;
  d_  = other.d_;
  Lt_ = other.Lt_;
  Lt_decomp_ = other.Lt_decomp_;
  Ut_decomp_ = other.Ut_decomp_;
  return *this;
};
*/




template< typename T >
template< typename S >
T gourd::nnp_hess<T>::vsqnorm_(
  const typename gourd::nnp_hess<T>::vector_type& v
) const {
  return v.squaredNorm();
};


template<>
template< typename S >
float gourd::nnp_hess<float>::vsqnorm_(
  const typename gourd::nnp_hess<float>::vector_type& v
) const {
  S vi, pa, pb;
  S err = 0;
  S ssq = 0;
  for ( int i = 0; i < v.size(); i++ ) {
    vi = static_cast<S>( v.coeffRef(i) );
    pa = vi * vi - err;
    pb = ssq + pa;
    err = (pb - ssq) - pa;
    ssq = pb;
  }
  return ssq;
};



template< typename T >
template< typename S >  /* Accumulator type */
double gourd::nnp_hess<T>::ldet() const {
  S ldi, pa, pb;
  S err = 0;
  S logdet = 0;
  for ( int i = 0; i < d_.size(); i++ ) {
    ldi = std::log(static_cast<S>( d_.coeff(i) ));
    pa = ldi - err;
    pb = logdet + pa;
    err = (pb - logdet) - pa;
    logdet = pb;
  }
  return logdet;
};




template< typename T >
template< typename S >
double gourd::nnp_hess<T>::qf(
  const typename gourd::nnp_hess<T>::vector_type& v
) const {
  return vsqnorm_<S>( d_.cwiseSqrt().asDiagonal() * (Lt_ * v) );
};



template< typename T >
template< typename S >
double gourd::nnp_hess<T>::iqf(
  const typename gourd::nnp_hess<T>::vector_type& v
) const {
  return vsqnorm_<S>( d_.cwiseSqrt().cwiseInverse().asDiagonal() *
		      Ut_decomp_.solve(v) );
};


template< typename T >
template< typename S >
double gourd::nnp_hess<T>::trqf(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  /* Trace of quadratic form:  m' Lt' D Lt m
   * tr(m' Lt' D Lt m) = tr(D^1/2 Lt m m' Lt' D^1/2)
   */
  const mat_type temp = d_.cwiseSqrt().asDiagonal() * (Lt_ * m);
  double trace = 0;
  for ( int j = 0; j < temp.cols(); j++ )
    trace += vsqnorm_<S>( temp.col(j) );
  return trace;
};




template< typename T >
template< typename S >
double gourd::nnp_hess<T>::triqf(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  /* Trace of quadratic form with inverse Hessian:
   * tr(m' Lt^-1 D^-1 Ut^-1 m)
   */
  const mat_type temp = d_.cwiseSqrt().cwiseInverse().asDiagonal() *
    (Ut_decomp_.solve(m));
  double trace = 0;
  for ( int j = 0; j < temp.cols(); j++ )
    trace += vsqnorm_<S>( temp.col(j) );
  return trace;
};




template< typename T >
int gourd::nnp_hess<T>::rank() const {
  return d_.size();
};


template< typename T >
typename gourd::nnp_hess<T>::mat_type
gourd::nnp_hess<T>::hprod(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  return Lt_.adjoint() * (d_.cwiseSqrt().asDiagonal() * m);
};



template< typename T >
typename gourd::nnp_hess<T>::mat_type
gourd::nnp_hess<T>::rmul(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  return Lt_.adjoint() * (d_.asDiagonal() * (Lt_ * m));
};



template< typename T >
typename gourd::nnp_hess<T>::mat_type
gourd::nnp_hess<T>::ihprod(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  return Lt_decomp_.solve(
    d_.cwiseSqrt().cwiseInverse().asDiagonal() * m
    );
};



template< typename T >
typename gourd::nnp_hess<T>::mat_type
gourd::nnp_hess<T>::irmul(
  const typename gourd::nnp_hess<T>::mat_type& m
) const {
  return Lt_decomp_.solve(
      d_.cwiseInverse().asDiagonal() * Ut_decomp_.solve(m)
    );
};


template< typename T >
typename gourd::nnp_hess<T>::spmat_type
gourd::nnp_hess<T>::hessian() const {
  return Lt_.adjoint() * d_.asDiagonal() * Lt_;
};


#endif  // _GOURD_NEAREST_NEIGHBOR_PROCESS_
