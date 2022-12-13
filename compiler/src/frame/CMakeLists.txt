file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/compiler/include/frame/*.h"
)
set(
    sources
    init.cpp
    edsl.cpp
)

create_cpp_library(
    NAME compiler_frame_lib
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/compiler/include/frame
    LINK_LIBS
    MLIRAADialectLib
    MLIRIR
    MLIRInferTypeOpInterface
    COPTS /EHsc
)
