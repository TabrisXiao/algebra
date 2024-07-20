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
    libs/builtin/normalization.cpp
    libs/math/algebra/ops.cpp
    libs/math/functional/ops.cpp
    libs/math/functional/passes.cpp
    libs/math/functional/analytic_derivative.cpp
    libs/SIO/exporter.cpp
)

create_cpp_library(
    NAME lgf_lib
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS ${COMPILER_OPTIONS}
)