
#include <string>
#include <vector>

#include "nifti2_io.h"
#include "afni_xml_io.h"


#ifndef _GOURD_CIFTI_XML_
#define _GOURD_CIFTI_XML_


namespace gourd {

  struct cifti_indices;
  struct cifti_dim_info;

  struct cifti_info {
    std::vector<int> VolumeDimensions;
    std::vector<gourd::cifti_indices> BrainModel;
    std::vector<gourd::cifti_dim_info> MatrixIndicesMap;
  };


  /*! Parsed CIFTI MatrixIndicesMap information
   *
   * Member names keep with the CIFTI field names
   */
  struct cifti_indices {
    int IndexCount;
    int IndexOffset;
    int SurfaceNumberOfVertices;
    int AppliesToMatrixDimension;
    std::string BrainStructure;
    std::string ModelType;
    std::vector<int> ind;

    cifti_indices() :
      IndexCount(0), IndexOffset(0), SurfaceNumberOfVertices(0),
      AppliesToMatrixDimension(0)
    { ; }
  };


  struct cifti_dim_info {
    double SeriesStart;
    double SeriesStep;
    int AppliesToMatrixDimension;
    int NumberOfSeriesPoints;
    std::string IndicesMapToDataType;
    std::string SeriesUnit;
  
    cifti_dim_info() :
      SeriesStart(0), SeriesStep(0),
      AppliesToMatrixDimension(0), NumberOfSeriesPoints(0)
    { ; }
  };



  

  /*! Parse CIFTI xml Data
   * 
   * Converts CIFTI xml extensions into a \c gourd::cifti_info
   * object. CIFTI inidices, for example, are then exposed as
   * as STL vectors of integers.
   */
  void parse_cifti_xml(
    const ::afni_xml_t* const xml,
    gourd::cifti_info& cinfo
  );


  
  void display( const gourd::cifti_info& ci );
  void display( const gourd::cifti_indices& ci );
  void display( const gourd::cifti_dim_info& cdi );

  
  /*! Display CIFTI xml information
   * 
   * Print xml information to stdout without parsing or
   * any manner of conversion
   */
  void display_cifti_xml(
    const ::afni_xml_t* const xml,
    const int depth = 0
  );


  
  void parse_cifti_attributes(
    const ::afni_xml_t* const xml,
    gourd::cifti_indices& cifi
  );
  
  void parse_cifti_attributes(
    const ::afni_xml_t* const xml,
    gourd::cifti_dim_info& cifi
  );
  
  void parse_cifti_volume_dimension(
    const char* const vold,
    std::vector<int>& vi
  );
  


  void parse_cifti_xml_vertex(
    const ::afni_xml_t* const xml,
    gourd::cifti_info& ci
  );

  void parse_cifti_xml_voxelijk(
    const ::afni_xml_t* const xml,
    gourd::cifti_info& ci
  );


  
};


#endif  // _GOURD_CIFTI_XML_


