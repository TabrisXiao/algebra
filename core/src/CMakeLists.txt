file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/core/include/*.h"
)
set(
    sources
    dgraph.cpp
    aog.cpp
)

create_cpp_library(
    NAME logic_abstract_lib
    HDRS
    ${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/core/include/
    COPTS /EHsc
)