
#include <stdexcept>
#include <vector>

#include "gourd/data/gplm_sstat.hpp"
#include "gourd/utilities/csv_reader.hpp"
#include "gourd/utilities/ragged_array.hpp"


#ifndef _GOURD_GPLMIX_SSTAT_
#define _GOURD_GPLMIX_SSTAT_


namespace gourd {

  /* ****************************************************************/
  /*! Sufficient statistics for gourd gaussian process mixed effects
   *  regression 
   */
  template< typename T >
  class gplmix_sstat : public gourd::gplm_sstat<T> {
  public:

    gplmix_sstat(
      const std::string xfile,
      const std::vector<std::string>& yfiles,
      const std::string surf_file,
      const gourd::ragged_array<int>& indices
    );

    const gourd::ragged_array<int>& varcomp_indices() const;

  private:
    gourd::ragged_array<int> vc_indices_;

    bool validate_indices_() const;
  };
  // class gplmix_sstat
  
};
// namespace gourd




template< typename T >
gourd::gplmix_sstat<T>::gplmix_sstat(
  const std::string xfile,
  const std::vector<std::string>& yfiles,
  const std::string surf_file,
  const gourd::ragged_array<int>& indices
) :
  gourd::gplm_sstat<T>(xfile, yfiles, surf_file)
{
  if ( indices.empty() ) {
    // Default: each coefficient gets its own variance component
    vc_indices_.resize( this->p() );
    for ( size_t i = 0; i < vc_indices_.size(); i++ )
      vc_indices_[i] = std::vector<int>(1u, (int)i);
  }
  else {
    // Copy indices and validate
    vc_indices_ = indices;
    if ( !validate_indices_() ) {
      throw std::domain_error("Invalid variance component indices");
    }
  }
};



template< typename T >
const gourd::ragged_array<int>&
gourd::gplmix_sstat<T>::varcomp_indices() const {
  return vc_indices_;
};


template< typename T >
bool gourd::gplmix_sstat<T>::validate_indices_() const {
  const int p = this->x().cols();
  std::vector<int> counts( p, 0 );
  bool ok = true;
  for ( auto& row : vc_indices_ ) {
    for ( int j : row ) {
      if (j < 0 || j >= p)  return false;
      counts[j]++;
    }
  }
  for ( int c : counts ) if ( c != 1 ) { ok = false; break; }
  if ( !ok ) {
    std::cerr << "Variance component errors:\n";
    for ( size_t i = 0; i < counts.size(); i++ ) {
      if ( counts[i] != 1 ) {
	std::cerr << "Index (" << i << ") used " << counts[i]
		  << " times\n";
      }
    } 
  }
  return ok;
};



#endif  // _GOURD_GPLMIX_SSTAT_
