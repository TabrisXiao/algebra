file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/codegen/*.h"
    "${PROJECT_SOURCE_DIR}/lgf/include/codegen/sketch/*.h"
)
set(
    sources
    ast.cpp
    cppWriter.cpp
    sketch/sketchLexer.cpp
)

create_cpp_library(
    NAME lgf_codegen_lib
    #HDRS
    #${headers}
    SRCS
    ${sources}
    LINK_LIBS
    lgf_lib
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS /EHsc
)
