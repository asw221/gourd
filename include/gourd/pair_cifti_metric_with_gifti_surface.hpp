
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "abseil/coordinates.hpp"
#include "abseil/utilities/string.hpp"

#include "afni_xml.h"
#include "afni_xml_io.h"
#include "gifti_io.h"
#include "nifti1.h"

#include "gourd/cifti_xml.hpp"


#ifndef _GOURD_PAIR_CIFTI_METRIC_WITH_GIFTI_SURFACE_
#define _GOURD_PAIR_CIFTI_METRIC_WITH_GIFTI_SURFACE_


namespace gourd {


  struct gifti_info {
    int index;
    int vertices;
    std::string structure;
    std::string geom_type;
    std::string topo_type;

    gifti_info() : index(-1), vertices(0)
    { ; }

    gifti_info( const gifti_info& other ) :
      index(other.index),
      vertices(other.vertices),
      structure(other.structure),
      geom_type(other.geom_type),
      topo_type(other.topo_type)
    { ; }
  };

  
  /* ****************************************************************/
  /*! Matched features of CIFTI and GIFTI files
   */
  class cifti_gifti_pair {
  public:
    cifti_gifti_pair() :
      paired_(false), cifti_array_dim_(0), brain_mod_index_(0)
    { ; }

    cifti_gifti_pair(
      ::nifti_image* nim,
      ::gifti_image* gim,
      const bool print_img_intent = false
    );

    operator bool() const;
    bool operator!() const;

    const cifti_indices&    brain_model() const;
    const std::vector<int>& cifti_paired_indices() const;
    const cifti_info&       cifti_paired_info() const;
    const std::vector<int>& gifti_indices() const;
    const gifti_info&       gifti_paired_info() const;

    int cifti_array_dim() const;
    
  private:
    /* Indicate if successful pairing was found */
    bool paired_;

    /* Dimension of CIFTI data array that corresponds with anatomical 
     * coordinates */
    int cifti_array_dim_;
    
    /* Index of paired CIFTI BrainModel */
    size_t brain_mod_index_;
    
    /* CIFTI index and dimension descriptors */
    struct cifti_info cim_info_;

    /* GIFTI anatomical dimension descriptors */
    struct gifti_info gim_info_;

    /* Indices of CIFTI data array that match GIFTI specs */
    std::vector<int> paired_cifti_indices_;
  };
  /* ****************************************************************/

  // size_t cim_stride;    /* CIFTI data array stride */
  

  void display( const gourd::gifti_info& ginfo ) {
    std::cout << "GIFTI Meta-data:\n"
	      << "  - index     : " << ginfo.index << "\n"
	      << "  - vertices  : " << ginfo.vertices << "\n"
	      << "  - structure : " << ginfo.structure << "\n"
	      << "  - geometry  : " << ginfo.geom_type << "\n"
	      << "  - topology  : " << ginfo.topo_type << "\n"
	      << std::endl;
  };
  
  /* 
   *   - Will try to not throw
   *   - Default to LEFT hemisphere and 0 dimensions
   *   - Assume \c gifti_image input is a surface file and has been
   *     pre-screened for boarding
   */
  gourd::cifti_gifti_pair pair_cifti_with_gifti(
    ::nifti_image* cim,
    ::gifti_image* gim
  );



  gourd::gifti_info gifti_parse_meta_anat( ::gifti_image* gim );

  

  template< typename CoordT >
  void extract_coordinates(
    const ::gifti_image* const gim,
    const gourd::cifti_gifti_pair& cgp,
    std::vector<
      abseil::cartesian_coordinate<3, CoordT>
      >& coord  
  );


  

  namespace def {
    
    template< typename ImT, typename CoordT >
    void extract_coordinates(
      const ::gifti_image* const gim,
      const gourd::cifti_gifti_pair& cgp,
      std::vector<
        abseil::cartesian_coordinate<3, CoordT>
        >& coord
    );

    
  };
  // namespace def


};
// namespace gourd









gourd::gifti_info gourd::gifti_parse_meta_anat( ::gifti_image* gim ) {
  gourd::gifti_info ginfo;
  bool structure_found = false;
  if ( gim->numDA > 0 && gim->darray ) {
    int i = 0;
    while ( i < gim->numDA && !structure_found ) {
      if ( gim->darray[i]->meta.length > 0 &&
	   gim->darray[i]->meta.name &&
	   gim->darray[i]->meta.value ) {
	for ( int j = 0; j < gim->darray[i]->meta.length; j++ ) {
	  std::string name(gim->darray[i]->meta.name[j]);
	  if ( name == "AnatomicalStructurePrimary" ) {
	    structure_found = true;
	    ginfo.structure = std::string(gim->darray[i]->meta.value[j]);
	    ginfo.index = i;
	    ginfo.vertices = gim->darray[i]->dims[0];
	  }
	  else if ( name == "GeometricType" ) {
	    ginfo.geom_type = std::string(gim->darray[i]->meta.value[j]);
	  }
	  else if ( name == "TopologicalType" ) {
	    ginfo.topo_type = std::string(gim->darray[i]->meta.value[j]);
	  }
	}  // for ( int j = 0; j < gim->darray[i]->meta.length; j++ )
      }  // if ( gim->darray[i]->meta.length > 0 && ...
      i++;
    }  // while ( i < gim->numDA && !structure_found )
  }  // if ( gim->numDA > 0 && gim->darray )
  if ( !structure_found ) {
    std::cerr << "\t*** Warning: 'AnatomicalStructurePrimary' not found "
	      << "in GIFTI file\n"
	      << "\t(gifti_parse_meta_anat)"
	      << std::endl;
  }
  return ginfo;
};









template< typename CoordT >
void gourd::extract_coordinates(
  const ::gifti_image* const gim,
  const gourd::cifti_gifti_pair& cgp,
  std::vector<
    abseil::cartesian_coordinate<3, CoordT>
    >& coord
) {
  if ( !gim->darray || gim->numDA <= 0 ) {
    throw std::domain_error(
      "extract_coordinates: surface ROI not included" );
  }
  if ( gim->darray[0]->intent != NIFTI_INTENT_POINTSET ) {
    std::cerr << "extract_coordinates: "
	      << "Unrecognized image intent code: "
	      << gim->darray[0]->intent << "\n";
  }
  /*
   * See gifti_convert_to_float( gifti_image * gim )
   * in file gifti_io.h/c
   */
  gourd::def::extract_coordinates<float>( gim, cgp, coord );
};



template< typename ImT, typename CoordT >
void gourd::def::extract_coordinates(
  const ::gifti_image* const gim,
  const gourd::cifti_gifti_pair& cgp,
  std::vector<
    abseil::cartesian_coordinate<3, CoordT>
    >& coord
) {
  /*
   * *** Use gifti_info here to make certain of darray index?
   */
  if ( gim->darray[0]->num_dim != 2 ) {
    throw std::domain_error(
      "extract_spherical_coordinates: requires xyz pointset" );
  }
  if ( gim->darray[0]->dims[1] != 3 ) {
    throw std::domain_error(
      "extract_spherical_coordinates: requires xyz pointset (2)" );
  }
  const ImT* data =
    static_cast<ImT*>( gim->darray[0]->data );
  const int nloc = gim->darray[0]->dims[0];
  const int ndim = gim->darray[0]->dims[1];
  const std::vector<int>& vertex_indices = cgp.gifti_indices();
  if ( ndim != 3 ) {
    std::cerr << "Encountered unexpected coordinate dimension ("
	      << ndim << ") in GIFTI file\n";
    throw std::domain_error("extract_coordinates: "
			    "unexpected coordinate dimension");
  }
  if ( vertex_indices.back() > nloc ) {
    std::cerr << "Encountered unexpected number of coordinates ("
	      << nloc << ") in GIFTI file\n";
    throw std::domain_error("extract_coordinates: "
			    "unexpected number of coordinates");
  }
  
  int stride = 0;
  std::array<CoordT, 3> xyz_buff;
  coord.clear();
  coord.reserve( vertex_indices.size() );
  for ( int i : vertex_indices ) {
    stride = i * 3;
    for ( int j = 0; j < 3; j++ ) {
      xyz_buff[j] = static_cast<CoordT>( *(data + stride) );
      stride++;
    }
    coord.push_back( abseil::cartesian_coordinate<3, CoordT>(
      xyz_buff.cbegin()) );
  }
};






/* ******************************************************************/
gourd::cifti_gifti_pair::cifti_gifti_pair(
  ::nifti_image* cim,
  ::gifti_image* gim,
  const bool print_img_intent
) {
  paired_ = true;
  cifti_array_dim_ = 0;
  brain_mod_index_ = 0;
  
  ::afni_xml_t* cim_xml = ::axio_cifti_from_ext( cim );
  gourd::parse_cifti_xml( cim_xml, cim_info_ );
  gim_info_ = gourd::gifti_parse_meta_anat( gim );
  ::axml_free_xml_t( cim_xml );
  
  // print image intents
  if ( print_img_intent ) {
    std::cerr << "CIFTI intent: "
	      << ::nifti_intent_string( cim->intent_code )
	      << "\n";

    std::cerr << "GIFTI intent:  { ";
    if ( gim->numDA > 0 && gim->darray ) {
      for ( int i = 0; i < gim->numDA; i++ ) {
	std::cerr << ::nifti_intent_string( gim->darray[i]->intent )
		  << ", ";
      }
    }
    std::cerr << "}\n";
  }
  //
  
  if ( cim_info_.BrainModel.empty() ) {
    throw std::domain_error("pair_cifti_with_gifti: insufficient "
			    "meta data in CIFTI file");
  }
  if ( gim_info_.index == -1 ) {
    throw std::domain_error("pair_cifti_with_gifti: insufficient "
			    "meta data in GIFTI file");
  }
 
  // Find related data dimensions
  std::string hemisphere = "LEFT";
  if ( abseil::find_ignore_case(gim_info_.structure, "Right") !=
       std::string::npos ) {
    hemisphere = "RIGHT";
  }
  else if ( abseil::find_ignore_case(
              gim_info_.structure, "Left") == std::string::npos ) {
    std::cerr << "\t*** Warning: Hemisphere ID not found in GIFTI"
	      << " file.\n\t\tDefaulting to LEFT cortex."
	      << "\n\t\t(pair_cifti_with_gifti)"
	      << std::endl;
  }
  bool cortex = false, hemisphere_matches = false;
  size_t i = 0;
  while ( !(cortex && hemisphere_matches) &&
	  i < cim_info_.BrainModel.size() ) {
    cortex = abseil::find_ignore_case(
      cim_info_.BrainModel[i].BrainStructure, "CORTEX" ) !=
      std::string::npos;
    hemisphere_matches = abseil::find_ignore_case(
      cim_info_.BrainModel[i].BrainStructure, hemisphere) !=
      std::string::npos;
    i++;
  }
  if ( cortex && hemisphere_matches ) {
    brain_mod_index_ = i - 1;
  }
  else {
    std::cerr << "\t*** Warning: Did not find meta data matching "
	      << hemisphere << " cortex in CIFTI file."
	      << "\n\t\t(pair_cifti_with_gifti)"
	      << std::endl;
  }

  // Make sure data refer to the same number of vertices
  const int cnv =
    cim_info_.BrainModel[brain_mod_index_].SurfaceNumberOfVertices;
  const int gnv = gim_info_.vertices;
  if ( cnv != gnv ) {
    paired_ = false;
    std::cerr << "CIFTI vertices: " << cnv << "\n"
	      << "GIFTI vertices: " << gnv << "\n";
    throw std::domain_error("pair_cifti_with_gifti: data refer "
			    "to different tesselations of cortex");
  }
  //

  // Find anatomical CIFTI array dimension
  // const std::vector<int> cdims = gourd::nifti2::get_dims( nim );
  int nnsd = 0;  // Number of Non-Singleton Dimensions
  for ( size_t i = 0; i < cim_info_.MatrixIndicesMap.size(); i++ ) {
    if ( abseil::find_ignore_case(
           cim_info_.MatrixIndicesMap[i].IndicesMapToDataType,
	   "BRAIN_MODELS") != std::string::npos ) {
      cifti_array_dim_ = nnsd;
      break;
    }
    if ( cim_info_.MatrixIndicesMap[i].NumberOfSeriesPoints > 1 )
      nnsd++;
  }

  // Initialize consecutive CIFTI data array indices
  paired_cifti_indices_
    .resize(cim_info_.BrainModel[brain_mod_index_].IndexCount);
  std::iota( paired_cifti_indices_.begin(),
	     paired_cifti_indices_.end(),
	     cim_info_.BrainModel[brain_mod_index_].IndexOffset );
};




gourd::cifti_gifti_pair::operator bool() const  {
  return paired_;
};


bool gourd::cifti_gifti_pair::operator!() const {
  return !this->operator bool();
};


const gourd::cifti_indices&
gourd::cifti_gifti_pair::brain_model() const {
  return cim_info_.BrainModel[brain_mod_index_];
};


const std::vector<int>&
gourd::cifti_gifti_pair::cifti_paired_indices() const {
  return paired_cifti_indices_;
};


const gourd::cifti_info& gourd::cifti_gifti_pair::cifti_paired_info() const {
  return cim_info_;
};


const std::vector<int>&
gourd::cifti_gifti_pair::gifti_indices() const {
  return brain_model().ind;
};


const gourd::gifti_info& gourd::cifti_gifti_pair::gifti_paired_info() const {
  return gim_info_;
};


int gourd::cifti_gifti_pair::cifti_array_dim() const {
  return cifti_array_dim_;
};



#endif  // _GOURD_PAIR_CIFTI_METRIC_WITH_GIFTI_SURFACE_






  // // Find CIFTI stride
  // cgp.cim_stride = 0;
  // i = 0;
  // while ( i < cgp.cim_info.MatrixIndicesMap.size() &&
  // 	  abseil::find_ignore_case(
  //           cgp.cim_info.MatrixIndicesMap[i].IndicesMapToDataType,
  // 	    "BRAIN_MODELS") !=
  // 	  std::string::npos ) {
  //   cgp.cim_stride +=
  //     (unsigned)cgp.cim_info.MatrixIndicesMap[i].NumberOfSeriesPoints;
  //   i++;
  // }
  // cgp.cim_stride = (cgp.cim_stride < 1 ) ? 1 : cgp.cim_stride;






