
#include <algorithm>
#include <iterator>
#include <vector>


#ifndef _ABSEIL_QUANTILE_
#define _ABSEIL_QUANTILE_

namespace abseil {

  
  /*! Numerical quantiles
   * 
   * Quantiles by linear interpolation of empirical CDF 
   */
  template<
    typename InputIt,
    typename value_type =
      typename std::iterator_traits<InputIt>::value_type
    >
  std::vector<value_type> quantile(
    InputIt x_begin,
    InputIt x_end,
    InputIt p_begin,
    InputIt p_end
  ) {
    assert( std::distance(x_begin, x_end) > 0 );
    assert( std::distance(p_begin, p_end) > 0 );
    const size_t np = std::distance(p_begin, p_end);
    std::vector<value_type> sx( x_begin, x_end );  // sorted x
    std::vector<value_type> q( np );               // quantiles
    std::sort( sx.begin(), sx.end() );
    InputIt pit = p_begin;
    for ( size_t i = 0; i < q.size(); ++i, ++pit ) {
      value_type p =
	std::max(std::min(*pit, (value_type)1), (value_type)0);
      size_t j = sx.size() * p;
      value_type w  = sx.size() * p - j;
      q[i] = (1 - w) * sx[j] + w * sx[j+1];
    }
    return q;
  };

  
}  // namespace abseil


#endif  // _ABSEIL_QUANTILE_
