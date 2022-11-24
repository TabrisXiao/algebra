
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tblgen.cmake)

set(TARGET_NAME "mlir_xsymbolic_compiler")
set(tblgen_src_base "tblgen")
set(tblgen_hdrs_output "include")
set(tblgen_docs_output "docs/generated")

add_dialect("MLA")
file(GLOB_RECURSE TBGLEN_SOURCE "*.td")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TBGLEN_SOURCE})
add_custom_target(TARGET_NAME ALL
    DEPENDS ${TBLGEN_OUTPUT_FILE}
)
#set_target_properties(${})
file(GLOB_RECURSE OP_LIB_SRCS "*.cpp")
file(GLOB_RECURSE OP_LIB_HDRS "*.hpp")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${OP_LIB_HDRS} ${OP_LIB_SRCS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_libs GLOBAL PROPERTY MLIRIR)

create_cpp_library(
    NAME mlir_compiler_object_lib
    HDRS
    ${OP_LIB_HDRS}
    SRCS
    ${OP_LIB_SRCS}
    INCLUDES
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    COPTS /W1 /std:c++17 /EHsc
    DEPS
    ${TARGET_NAME}
    ${dialect_libs}
    ${conversion_libs}
)