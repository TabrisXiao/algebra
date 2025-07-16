file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/graph/*.h"
    #"${PROJECT_SOURCE_DIR}/lgf/include/libs/LinearAlg/*.h"
)

set(
    sources
    object.cpp
    operation.cpp
    graph.cpp
)

create_cpp_library(
    NAME lgf_graph_core
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    LINK_LIBS
    utils_lib
    COPTS ${COMPILER_OPTIONS}
)