
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