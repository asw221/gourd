
## include(FindEXPAT)
##
find_package( EXPAT REQUIRED )

set(NIFTI_CIFTILIB_NAME ${NIFTI_PACKAGE_PREFIX}cifti)


set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )


add_library(${NIFTI_CIFTILIB_NAME} afni_xml.c afni_xml_io.c )
## set_target_properties(
##   ${NIFTI_CIFTILIB_NAME}
##   PROPERTIES
##     PRIVATE_HEADER  "${CMAKE_CURRENT_LIST_DIR}/afni_xml.h;${CMAKE_CURRENT_LIST_DIR}/afni_xml_io.h"
##     )
## target_link_libraries( ${NIFTI_CIFTILIB_NAME} PUBLIC EXPAT::EXPAT
## ${NIFTI_PACKAGE_PREFIX}nifti2)
##
target_link_libraries( ${NIFTI_CIFTILIB_NAME} PUBLIC EXPAT::EXPAT
		       NIFTI::nifti2 )
##
# Set lib version when buildung shared libs.
if(BUILD_SHARED_LIBS)
    set_target_properties(${NIFTI_CIFTILIB_NAME} PROPERTIES ${NIFTI_LIBRARY_PROPERTIES})
endif()
## install_nifti_target(${NIFTI_CIFTILIB_NAME})


target_include_directories( ${NIFTI_CIFTILIB_NAME} PUBLIC
			    "${NIFTI_SOURCE_DIR}/include"
			    "${NIFTI_SOURCE_DIR}/nifti2"
			    "${NIFTI_SOURCE_DIR}/cifti" )




add_library( NIFTI::cifti ALIAS ${NIFTI_CIFTILIB_NAME} )


## ---


if(NIFTI_BUILD_APPLICATIONS)
  add_nifti_executable(${NIFTI_PACKAGE_PREFIX}afni_xml_tool afni_xml_tool.c)
  target_link_libraries(${NIFTI_PACKAGE_PREFIX}afni_xml_tool PUBLIC ${NIFTI_CIFTILIB_NAME})
  add_nifti_executable(${NIFTI_PACKAGE_PREFIX}cifti_tool cifti_tool.c)
  target_link_libraries(${NIFTI_PACKAGE_PREFIX}cifti_tool PUBLIC ${NIFTI_CIFTILIB_NAME})
  install_nifti_target(${NIFTI_PACKAGE_PREFIX}afni_xml_tool)
  install_nifti_target(${NIFTI_PACKAGE_PREFIX}cifti_tool)
endif()
