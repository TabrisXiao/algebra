file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/lgf/*.h"
    "${PROJECT_SOURCE_DIR}/lgf/include/libs/*.h"
    #"${PROJECT_SOURCE_DIR}/lgf/include/libs/LinearAlg/*.h"
)

set(
    sources
    builtin/normalization.cpp
    math/algebra/ops.cpp
    math/normalization.cpp
    math/functional/ops.cpp
    math/functional/passes.cpp
    math/functional/analytic_derivative.cpp
    SIO/exporter.cpp
)

create_cpp_library(
    NAME lgf_lib
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    ${PROJECT_SOURCE_DIR}/lgf/include/libs/generated/
    LINK_LIBS
    lgf_core
    COPTS ${COMPILER_OPTIONS}
)