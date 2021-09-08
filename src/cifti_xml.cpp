

#ifndef _GOURD_CIFTI_XML_CPP_
#define _GOURD_CIFTI_XML_CPP_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nifti2_io.h"
#include "afni_xml_io.h"

#include "abseil/utilities/string.hpp"

#include "gourd/cifti_xml.hpp"




std::string gourd::brainmodel_xml(
  const gourd::cifti_indices& model,
  const bool ltag,
  const bool rtag
) {
  std::ostringstream ss;
  if ( ltag )  ss << "<BrainModel ";
  ss << "IndexOffset=\"" << model.IndexOffset << "\" "
     << "IndexCount=\"" << model.IndexCount << "\" "
     << "BrainStructure=\"" << model.BrainStructure << "\" "
     << "ModelType=\"" << model.ModelType << "\" "
     << "SurfaceNumberOfVertices=\"" << model.SurfaceNumberOfVertices << "\">"
     << "\n" << std::string(16, ' ') << "<VertexIndices>";
  const size_t n = model.ind.size();
  for ( size_t i = 0; i < n; i++ ) {
    ss << i;
    if ( i < (n-1) ) ss << " ";
  }
  ss << "</VertexIndices>\n";
  if ( rtag )  ss << std::string(12, ' ') << "</BrainModel>";
  return ss.str();
};




void gourd::display(
  const gourd::cifti_info& ci
) {
  // Volume Dimension info first
  if ( !ci.VolumeDimensions.empty() ) {
    std::cout << "Volume Dim: (";
    for ( size_t i = 0; i < ci.VolumeDimensions.size(); i++ ) {
      std::cout << ci.VolumeDimensions[i] << "x";
    }
    std::cout << "\b)\n" << std::endl;
  }

  
  // MatrixIndices info
  if ( !ci.MatrixIndicesMap.empty() ) {
    for ( size_t i = 0; i < ci.MatrixIndicesMap.size(); i++ ) {
      std::cout << "Matrix Indices Map {" << i << "}\n"
		<< "\tApplies to Dim   :  "
		<< ci.MatrixIndicesMap[i].AppliesToMatrixDimension << "\n"
		<< "\tMapped Data Type :  "
		<< ci.MatrixIndicesMap[i].IndicesMapToDataType << "\n"
		<< "\tSeries Length    :  "
		<< ci.MatrixIndicesMap[i].NumberOfSeriesPoints << "\n"
		<< "\tSeries Units     :  "
		<< ci.MatrixIndicesMap[i].SeriesUnit << "\n"
		<< "\tSeries Start     :  "
		<< ci.MatrixIndicesMap[i].SeriesStart << "\n"
		<< "\tSeries Step      :  "
		<< ci.MatrixIndicesMap[i].SeriesStep << "\n"
		<< std::endl;
    }
    std::cout << std::endl;
  }

  
  // Brain Model info
  if ( !ci.BrainModel.empty() ) {
    for ( size_t i = 0; i < ci.BrainModel.size(); i++ ) {
      std::cout << "Brain Model {" << i << "}  :  "
		<< ci.BrainModel[i].ind.size() << "\n"
		<< "\tIndex Offset     :  "
		<< ci.BrainModel[i].IndexOffset << "\n" 
		<< "\tIndex Count      :  "
		<< ci.BrainModel[i].IndexCount << "\n" 
		<< "\tBrain Structure  :  "
		<< ci.BrainModel[i].BrainStructure << "\n" 
		<< "\tModel Type       :  "
		<< ci.BrainModel[i].ModelType << "\n" 
		<< "\tSurface Vertices :  "
		<< ci.BrainModel[i].SurfaceNumberOfVertices << "\n"
		<< std::endl;
    }
    std::cout << std::endl;
  }
};



void gourd::display( const gourd::cifti_indices& ci ) {
  std::cout << "Index Offset     :  "
	    << ci.IndexOffset << "\n" 
	    << "Index Count      :  "
	    << ci.IndexCount << "\n" 
	    << "Brain Structure  :  "
	    << ci.BrainStructure << "\n" 
	    << "Model Type       :  "
	    << ci.ModelType << "\n" 
	    << "Surface Vertices :  "
	    << ci.SurfaceNumberOfVertices << "\n"
	    << std::endl;  
};


void gourd::display( const cifti_dim_info& cdi ) {
  std::cout << "Applies to Dim   :  "
	    << cdi.AppliesToMatrixDimension << "\n"
	    << "Mapped Data Type :  "
	    << cdi.IndicesMapToDataType << "\n"
	    << "Series Length    :  "
	    << cdi.NumberOfSeriesPoints << "\n"
	    << "Series Units     :  "
	    << cdi.SeriesUnit << "\n"
	    << "Series Start     :  "
	    << cdi.SeriesStart << "\n"
	    << "Series Step      :  "
	    << cdi.SeriesStep << "\n"
	    << std::endl;
};




void gourd::display_cifti_xml(
  const ::afni_xml_t* const xml,
  const int depth
) {
  const std::string name( xml->name );
  const std::string indent(depth, '\t');
  const int nchild = xml->nchild;
  std::ostringstream ssnm;

  ssnm << indent << "* " << name << " {" << nchild << "}";
  std::cout << ssnm.str() << std::endl;

  if ( xml->attrs.length > 0 && xml->attrs.name && xml->attrs.value) {
    for ( int i = 0; i < xml->attrs.length; i++ ) {
      std::ostringstream ss;
      ss << indent << "  - "
	 << xml->attrs.name[i] << ":  "
	 << abseil::trim(xml->attrs.value[i]);
      std::cout << ss.str() << std::endl;
    }
    std::cout << std::endl;
  }

  if ( xml->nchild > 0 && xml->xchild ) {
    for ( int i = 0; i < xml->nchild; i++ )
      gourd::display_cifti_xml( xml->xchild[i], depth + 1 );
  }
};


void gourd::parse_cifti_xml(
  const ::afni_xml_t* const xml,
  gourd::cifti_info& cinfo
) {
  if ( xml->nchild > 0 && xml->xchild ) {
    const std::string name( xml->name );
    for ( int i = 0; i < xml->nchild; i++ ) {
      const std::string child_name( xml->xchild[i]->name );
      if ( child_name == "MatrixIndicesMap" ) {
	cinfo.MatrixIndicesMap.emplace_back();
	gourd::parse_cifti_attributes( xml->xchild[i],
				       cinfo.MatrixIndicesMap.back() );
      }
      if ( name == "BrainModel" ) {
	cinfo.BrainModel.emplace_back();
	gourd::parse_cifti_attributes( xml, cinfo.BrainModel.back() );	
    	if ( child_name == "VertexIndices" ) {
    	  gourd::parse_cifti_xml_vertex( xml->xchild[i], cinfo );
    	}
    	else if ( child_name == "VoxelIndicesIJK" ) {
    	  gourd::parse_cifti_xml_voxelijk( xml->xchild[i], cinfo );
    	}
      }
      else if ( name == "Volume" ) {
	for ( int j = 0; j < xml->attrs.length; j++ ) {
	  if ( std::string(xml->attrs.name[j]) ==
	       "VolumeDimensions" ) {
	    gourd::parse_cifti_volume_dimension(
              xml->attrs.value[j], cinfo.VolumeDimensions );
	  }
	}
      }
      else {
	gourd::parse_cifti_xml( xml->xchild[i], cinfo );
      }
    }
  }  // if ( xml->nchild >= 1 )
};




void gourd::parse_cifti_xml_vertex(
  const ::afni_xml_t* const xml,
  gourd::cifti_info& ci
) {
  if ( xml->xlen > 0 && xml->xtext ) {

    // Process index information. (i,j,k) tuples are left as is for now
    ci.BrainModel.back().ind.reserve( xml->xlen / 2 );
    std::istringstream isi( xml->xtext );
    if ( isi ) {
      while ( isi ) {
	std::string atom;
	if ( std::getline(isi, atom, ' ') ) {
	  try {
	    ci.BrainModel.back().ind.push_back( std::stoi(atom) );
	  }
	  catch (...) { ; }
	}  // if ( std::getline(isi, segment) )
      }  // while ( isi )
    }  // if ( isi )

    ci.BrainModel.back().ind.shrink_to_fit();
  }  // if ( xml->xlen >= 1 )
};



void gourd::parse_cifti_xml_voxelijk(
  const ::afni_xml_t* const xml,
  gourd::cifti_info& ci
) {
  // The (i,j,k) and (i',j',k') indices can apparently be separated
  // by either newline characters or just spaces
  //
  // parse_cifti_xml_voxelijk will rescale (i,j,k) to flat
  // indices in column-major order if ci.VolumeDimensions
  // has been set. Otherwise, retains a long vector of
  // (i,j,k,   i',j',k'  ...) indices
  //
  if ( xml->xlen > 0 && xml->xtext ) {

    const int D = ((int)ci.VolumeDimensions.size() > 0) ?
      (int)ci.VolumeDimensions.size() : 1;
    
    int counter = 0;
    int A = 0, B = 0;
    int s = 1;
    int j = 0;
    std::vector<int> stride( D );

    if ( !ci.VolumeDimensions.empty() ) {
      for ( int i = 0; i < D; i++ ) {
	stride[i] = s;
	s *= ci.VolumeDimensions[i];
      }
    }
    else {
      stride[0] = s;
    }    

    // Process index information. (i,j,k) tuples are left as is for now
    ci.BrainModel.back().ind.reserve( xml->xlen / 2 );
    std::istringstream isi( xml->xtext );
    if ( isi ) {
      while ( isi ) {
	std::string segment;
	if ( std::getline(isi, segment) ) {
	  std::istringstream iseg( segment );
	  while ( iseg ) {
	    std::string atom;
	    if ( std::getline(iseg, atom, ' ') ) {
	      try {
		A = std::stoi(atom);
		j = counter % D;
		B += A * stride[j];
		counter++;
		if ( counter % D == 0 ) {
		  ci.BrainModel.back().ind.push_back( B );
		  B = 0;
		}
	      }
	      catch (...) { ; }
	    }
	  }  // while ( iseg )
	}  // if ( std::getline(isi, segment) )
      }  // while ( isi )
    }  // if ( isi )

    ci.BrainModel.back().ind.shrink_to_fit();
  }  // if ( xml->xlen >= 1 )
};
  




void gourd::parse_cifti_volume_dimension(
  const char* const vold,
  std::vector<int>& vi
) {
  std::istringstream ivol( vold );
  vi.clear();
  while ( ivol ) {
    std::string atom;
    if ( std::getline(ivol, atom, ',') ) {
      try {
	vi.push_back( std::stoi(atom) );
      }
      catch (...) { ; }
    }
  }
};



void gourd::parse_cifti_attributes(
  const ::afni_xml_t* const xml,
  gourd::cifti_indices& cifi
) {
  // Process additional attribute information
  if ( xml->attrs.length > 0 && xml->attrs.name && xml->attrs.value ) {
    for ( int i = 0; i < xml->attrs.length; i++ ) {
      try {
	const std::string attr_name( xml->attrs.name[i] );
	if ( attr_name == "IndexCount" ) {
	  cifi.IndexCount = std::stoi( xml->attrs.value[i] );
	}
	else if ( attr_name == "IndexOffset" ) {
	  cifi.IndexOffset = std::stoi( xml->attrs.value[i] );
	}
	else if ( attr_name == "SurfaceNumberOfVertices" ) {
	  cifi.SurfaceNumberOfVertices = std::stoi( xml->attrs.value[i] );
	}
	else if ( attr_name == "BrainStructure" ) {
	  cifi.BrainStructure = std::string( xml->attrs.value[i] );
	}
	else if ( attr_name == "ModelType" ) {
	  cifi.ModelType = std::string( xml->attrs.value[i] );
	}
      }
      catch ( const std::exception& ex ) {
	std::cerr << "\t\t*** Error Caught: " << ex.what() << " ***\n";
      }
      catch (...) { ; }
    }
  }  // if ( xml->attrs.length > 0 )  
};




void gourd::parse_cifti_attributes(
  const ::afni_xml_t* const xml,
  gourd::cifti_dim_info& cifi
) {
  // Process additional attribute information
  if ( xml->attrs.length > 0 && xml->attrs.name && xml->attrs.value ) {
    for ( int i = 0; i < xml->attrs.length; i++ ) {
      try {
	const std::string attr_name( xml->attrs.name[i] );
	if ( attr_name == "AppliesToMatrixDimension" ) {
	  cifi.AppliesToMatrixDimension = std::stoi( xml->attrs.value[i] );
	}
	else if ( attr_name == "IndicesMapToDataType" ) {
	  cifi.IndicesMapToDataType = std::string( xml->attrs.value[i] );
	}
	else if ( attr_name == "NumberOfSeriesPoints" ) {
	  cifi.NumberOfSeriesPoints = std::stoi( xml->attrs.value[i] );
	}
	else if ( attr_name == "SeriesStart" ) {
	  cifi.SeriesStart = std::stod( xml->attrs.value[i] );
	}
	else if ( attr_name == "SeriesStep" ) {
	  cifi.SeriesStep = std::stod( xml->attrs.value[i] );
	}
	else if ( attr_name == "SeriesUnit" ) {
	  cifi.SeriesUnit = std::string( xml->attrs.value[i] );
	}
      }
      catch ( const std::exception& ex ) {
	std::cerr << "\t\t*** Error Caught: " << ex.what() << " ***\n";
      }
      catch (...) { ; }
    }
  }
};




#endif  // _GOURD_CIFTI_XML_CPP_

