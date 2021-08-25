
project( bobyqa )
cmake_minimum_required( VERSION 3.0 )

# install destinations
set( ALG_INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin" )
set( ALG_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib"
)
set( ALG_INSTALL_INCLUDE_DIR
     "${CMAKE_INSTALL_PREFIX}/include/optim" )

set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib" )

set( BOBYQA_LIBRARY_PROPERTIES VERSION "0.0.0" )



set( BOBYQA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bobyqa.c" )

## Define SINGLE_PRECISION to use bobyqa with in single precision


# static version
add_library( bobyqa_static STATIC ${BOBYQA_SRC} )

set_target_properties( bobyqa_static
		       PROPERTIES ${BOBYQA_LIBRARY_PROPERTIES}
		       OUTPUT_NAME "bobyqa" )

target_include_directories( bobyqa_static PUBLIC
			    "${bobyqa_SOURCE_DIR}" )

add_library( OPTIM::bobyqa ALIAS bobyqa_static )
