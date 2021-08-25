
project(NIFTI)

cmake_minimum_required(VERSION 3.0)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

subdirs( znzlib niftilib nifti2 cifti )
add_definitions(-DHAVE_ZLIB)

# Add option to build shared libraries. The default is OFF to maintain the
# current build behavior
option( BUILD_SHARED_LIBS "Toggle building shared libraries")



###  USE AS STAND ALONE PACKAGE
find_package( ZLIB REQUIRED )




set(PACKAGE_PREFIX "")
  # Set default shared library version
  # This library version will be applied to all libraries in the package
  # unless it is not explicitely for a certain lib.
set( NIFTI_SHAREDLIB_VERSION
     ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
     )
string( REGEX MATCH "^[0-9]+" NIFTI_SHAREDLIB_SOVERSION ${NIFTI_SHAREDLIB_VERSION})
set( NIFTI_LIBRARY_PROPERTIES VERSION ${NIFTI_SHAREDLIB_VERSION}
     SOVERSION ${NIFTI_SHAREDLIB_SOVERSION} )

set( NIFTI_INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin" )
set( NIFTI_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib" )

set( NIFTI_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/nifti"
)

set( NIFTI_ZLIB_LIBRARIES ${ZLIB_LIBRARIES} )
set( NIFTI_INSTALL_NO_DEVELOPMENT 0 )
set( NIFTI_INSTALL_NO_LIBRARIES 0 )

# Needs an if clause and more work before testing can take place.
# subdirs(Testing)
##
subdirs(nifticdf)



option(USE_FSL_CODE "If OFF, The copyright of this code is questionable for inclusion with nifti." OFF)
mark_as_advanced(USE_FSL_CODE)

# The order of subdirs is important! fsliolob has to preceed examples
# as otherwise FSLIOLIB_SOURCE_DIR is undefined and hence the examples
# will fail to compile
if(USE_FSL_CODE)
  subdirs(fsliolib)
  subdirs(examples)
endif(USE_FSL_CODE)
# enable_testing()
# include(CTest)




if(NOT DEFINED NIFTI_INSTALL_LIB_DIR)
  message(FATAL_ERROR "Must set NIFTI_INSTALL_LIB_DIR.")
endif(NOT DEFINED NIFTI_INSTALL_LIB_DIR)

if(NOT DEFINED NIFTI_INSTALL_BIN_DIR)
  message(FATAL_ERROR "Must set NIFTI_INSTALL_BIN_DIR.")
endif(NOT DEFINED NIFTI_INSTALL_BIN_DIR)

if(NOT DEFINED NIFTI_INSTALL_INCLUDE_DIR)
  message(FATAL_ERROR "Must set NIFTI_INSTALL_INCLUDE_DIR.")
endif(NOT DEFINED NIFTI_INSTALL_INCLUDE_DIR)


