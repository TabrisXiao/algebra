file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/lgf/*.h"
    #"${PROJECT_SOURCE_DIR}/lgf/include/libs/LinearAlg/*.h"
)

set(
    sources
    code.cpp
    value.cpp
    edge.cpp
    node.cpp
    pass.cpp
)

create_cpp_library(
    NAME lgf_core
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS ${COMPILER_OPTIONS}
)