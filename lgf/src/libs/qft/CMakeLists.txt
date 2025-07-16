file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/graph/*.h"
    "${PROJECT_SOURCE_DIR}/lgf/include/libs/qft/*.h"
)

set(
    sources
    qft.cpp
)

create_cpp_library(
    NAME lgf_qft
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/libs/
    LINK_LIBS
    lgf_graph_core
    COPTS ${COMPILER_OPTIONS}
)