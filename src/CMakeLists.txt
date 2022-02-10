
find_package( Boost      REQUIRED )
find_package( Eigen3 3.3 REQUIRED NO_MODULE )
find_package( EXPAT      REQUIRED )
find_package( OpenMP     REQUIRED )
find_package( ZLIB       REQUIRED )


# find_package(FFTW REQUIRED)
# find_package(ITK REQUIRED)
# find_package(NLOPT REQUIRED)


if (NOT TARGET OpenMP::OpenMP_CXX)
  message( FATAL_ERROR "OpenMP_CXX target not found" )
endif()




include_directories( ${gourd_SOURCE_DIR}/include )
include_directories( ${gourd_SOURCE_DIR}/include/nifti )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${gourd_SOURCE_DIR}/lib/abseil/include )


link_directories(${gourd_SOURCE_DIR}/lib)




## --- hello_gourd ---------------------------------------------------
add_executable(hello_gourd hello_gourd.cpp)
set_target_properties(hello_gourd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")

## --- gourd_credband ------------------------------------------------
add_executable(gourd_credband gourd_credband.cpp cifti_xml.cpp)
set_target_properties(gourd_credband PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_credband
		NIFTI::cifti NIFTI::nifti2 NIFTI::gifti
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)


## --- gourd_covest --------------------------------------------------
add_executable(gourd_covest gourd_covest.cpp cifti_xml.cpp)
set_target_properties(gourd_covest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_covest
	OPTIM::bobyqa
	NIFTI::gifti NIFTI::cifti NIFTI::nifti2
	EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)


## --- gourd_gplm ----------------------------------------------------
add_executable(gourd_gplm gourd_gplm.cpp cifti_xml.cpp)
set_target_properties(gourd_gplm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)


## --- gourd_gplmix --------------------------------------------------
add_executable(gourd_gplmix gourd_gplmix.cpp cifti_xml.cpp)
set_target_properties(gourd_gplmix PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplmix
		OPTIM::bobyqa
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)



## --- gourd_gplmmm --------------------------------------------------
add_executable(gourd_gplmmm gourd_gplmmm.cpp cifti_xml.cpp)
set_target_properties(gourd_gplmmm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplmmm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)





## --- gourd_sglm ----------------------------------------------------
add_executable(gourd_sglm gourd_sglm.cpp cifti_xml.cpp)
set_target_properties(gourd_sglm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_sglm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)


## --- cifti_info ----------------------------------------------------
add_executable(cifti_info cifti_info.cpp
			  cifti_xml.cpp)
set_target_properties(cifti_info PROPERTIES RUNTIME_OUTPUT_DIRECTORY
		"${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(cifti_info NIFTI::cifti NIFTI::nifti2)




## --- gourd_vwise_glm -----------------------------------------------
add_executable(gourd_vwise_glm gourd_vwise_glm.cpp cifti_xml.cpp)
set_target_properties(gourd_vwise_glm PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_vwise_glm
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)







## --- gourd_gplm2 ---------------------------------------------------
## (Independent HMC momenta variant)
##
add_executable(gourd_gplm2 gourd_gplm.cpp cifti_xml.cpp)
set_target_properties(gourd_gplm2 PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
target_link_libraries(gourd_gplm2
		NIFTI::gifti NIFTI::cifti NIFTI::nifti2
		EXPAT::EXPAT ZLIB::ZLIB Eigen3::Eigen)
target_compile_definitions(gourd_gplm2
		PUBLIC GOURD_GPL_INDEPENDENT_MOMENTA)



# target_compile_definitions(___ PUBLIC EIGEN_DONT_PARALLELIZE)
# ## EIGEN_DONT_PARALLELIZE
# ## EIGEN_DONT_VECTORIZE


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
