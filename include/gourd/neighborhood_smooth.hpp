
#include <cassert>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "abseil/coordinates.hpp"
#include "abseil/covariance_functors.hpp"

#include "gourd/options.hpp"  // gourd::dist_code



#ifndef _GOURD_NEIGHBORHOOD_SMOOTH_
#define _GOURD_NEIGHBORHOOD_SMOOTH_

namespace gourd {

  template< typename T, size_t D >
  Eigen::SparseMatrix<T, Eigen::RowMajor> compute_nnsmooth_mat(
    const std::vector< abseil::cartesian_coordinate<3, T> >& coords,
    const abseil::covariance_functor<T, D>* const cov,
    const T r,
    const gourd::dist_code dc
  ) {
    assert(r > 0 && "compute_nnsmooth_mat: neighborhood radius <= 0");
    typedef typename Eigen::Triplet<T> triplet_type;
    using dist_c = gourd::dist_code;
    const T surf_radius = abseil::to_spherical(coords[0]).radius();
    const T maxcl = 2 * surf_radius * std::sin(r / (surf_radius * 2));
    const T nbr_ratio = (1 - std::cos(r / surf_radius)) / 2;
    const T compar = (dc == dist_c::great_circle) ? maxcl : r;
    //
    int reserve_n = coords.size() * coords.size() * nbr_ratio;
    int prev_nbrs = 1;
    std::vector<triplet_type> trips;
    trips.reserve( reserve_n );
    //
    for ( size_t i = 0; i < coords.size(); i++ ) {
      reserve_n = prev_nbrs * 11 / 10;
      std::vector<int> nbrs;  // neighborhood indices
      std::vector<T> weights; // smoothing weights
      T wsum = 0;
      nbrs.reserve( reserve_n );
      weights.reserve( reserve_n );
      for ( size_t j = 0; j < coords.size(); j++ ) {
	const T dx = std::abs( coords[j][0] - coords[i][0] );
	const T dy = std::abs( coords[j][1] - coords[i][1] );
	const T dz = std::abs( coords[j][2] - coords[i][2] );
	if ( dx <= maxcl && dy <= maxcl && dz <= maxcl ) {
	  const T chord = std::sqrt( dx*dx + dy*dy + dz*dz );
	  const T dist = (dc == dist_c::great_circle) ?
	    2 * surf_radius * std::asin( chord / (2 * surf_radius) ) :
	    chord;
	  if ( dist <= compar ) {
	    const T w = (*cov)(dist);
	    nbrs.push_back( j );
	    weights.push_back( w );
	    wsum += w;
	  }
	}
      }  // for ( size_t j = 0; j < coords.size(); j++ )
      nbrs.shrink_to_fit();
      weights.shrink_to_fit();
      //
      for ( size_t j = 0; j < nbrs.size(); j++ ) {
	weights[j] /= wsum;
	trips.push_back( triplet_type(i, nbrs[j], weights[j]) );
      }
      prev_nbrs = nbrs.size();
    }  // for ( size_t i = 0; i < coords.size(); i++ )
    //
    trips.shrink_to_fit();
    Eigen::SparseMatrix<T, Eigen::RowMajor> smooth(
      coords.size(), coords.size());
    smooth.setFromTriplets( trips.begin(), trips.end() );
    return smooth;
  };


}  // namespace gourd

#endif  // _GOURD_NEIGHBORHOOD_SMOOTH_

