
find_package( Boost QUIET )  # No longer REQUIRED

find_package( EXPAT      REQUIRED )
find_package( OpenMP     REQUIRED )
find_package( ZLIB       REQUIRED )
## find_package( Eigen3 3.3...<4 REQUIRED NO_MODULE )


if (NOT TARGET OpenMP::OpenMP_CXX)
  message( FATAL_ERROR "OpenMP_CXX target not found" )
endif()




include_directories( ${gourd_SOURCE_DIR}/include )
include_directories( ${gourd_SOURCE_DIR}/include/nifti )
include_directories( ${gourd_SOURCE_DIR}/lib/Eigen3 )
include_directories( ${gourd_SOURCE_DIR}/lib/abseil/include )

if ( DEFINED Boost_INCLUDE_DIRS )
  include_directories( ${Boost_INCLUDE_DIRS} )
endif()


link_directories( ${gourd_SOURCE_DIR}/lib )




## --- hello_gourd ---------------------------------------------------
add_executable(hello_gourd hello_gourd.cpp)
set_target_properties(hello_gourd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")


## --- gourd_credband ------------------------------------------------
## Compute credible bands from MCMC log files
##
add_executable(gourd_credband gourd_credband.cpp cifti_xml.cpp)
set_target_properties(gourd_credband PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_credband
		NIFTI::cifti NIFTI::nifti2 NIFTI::gifti
		EXPAT::EXPAT ZLIB::ZLIB)


## --- gourd_covest --------------------------------------------------
## Estimate spatial covariance function parameters
##
add_executable(gourd_covest gourd_covest.cpp cifti_xml.cpp)
set_target_properties(gourd_covest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_covest
	OPTIM::bobyqa
	NIFTI::gifti NIFTI::cifti NIFTI::nifti2
	EXPAT::EXPAT ZLIB::ZLIB)


## --- gourd_gplm ----------------------------------------------------
## Working model variant
##
add_executable(gourd_gplm gourd_gplm.cpp cifti_xml.cpp)
set_target_properties(gourd_gplm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)


## --- gourd_gplmme --------------------------------------------------
## Marginal effect variant (correlated error)
##
add_executable(gourd_gplmme gourd_gplmme.cpp cifti_xml.cpp)
set_target_properties(gourd_gplmme PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplmme
		OPTIM::bobyqa
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)



## --- gourd_gplmcm --------------------------------------------------
## Conditional effect variant (MAP estimates-based)
##
add_executable(gourd_gplmce gourd_gplmce.cpp cifti_xml.cpp)
set_target_properties(gourd_gplmce PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplmce
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)





## --- gourd_sglm ----------------------------------------------------
## Vertex-wise linear models with neighborhood-based spatial smoothing
##
add_executable(gourd_sglm gourd_sglm.cpp cifti_xml.cpp)
set_target_properties(gourd_sglm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_sglm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)


## --- cifti_info ----------------------------------------------------
add_executable(cifti_info cifti_info.cpp
			  cifti_xml.cpp)
set_target_properties(cifti_info PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(cifti_info NIFTI::cifti NIFTI::nifti2)




## --- gourd_vwise_glm -----------------------------------------------
## Vertex-wise linear models (no spatial smoothing)
##
add_executable(gourd_vwise_glm gourd_vwise_glm.cpp cifti_xml.cpp)
set_target_properties(gourd_vwise_glm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_vwise_glm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)







## --- gourd_gplm2 ---------------------------------------------------
## (Independent HMC momenta variant)
##
add_executable(gourd_gplm2 gourd_gplm.cpp cifti_xml.cpp)
set_target_properties(gourd_gplm2 PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplm2
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB)
  ## Eigen3::Eigen
target_compile_definitions(gourd_gplm2
		PUBLIC GOURD_GPL_INDEPENDENT_MOMENTA)




# target_compile_definitions(___ PUBLIC EIGEN_DONT_PARALLELIZE)
# ## EIGEN_DONT_PARALLELIZE
# ## EIGEN_DONT_VECTORIZE
##
## ## --- gourd_gplmixap ------------------------------------------------
## add_executable(gourd_gplmixap gourd_gplmix.cpp cifti_xml.cpp)
## set_target_properties(gourd_gplmixap PROPERTIES
## 		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
## target_link_libraries(gourd_gplmixap
## 		OPTIM::bobyqa
## 		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
## 		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)
## target_compile_definitions(gourd_gplmixap
## 		PUBLIC GOURD_GPLMIX_NNGP_APPROX)
