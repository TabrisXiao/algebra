file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/lgf/*.h"
    "${PROJECT_SOURCE_DIR}/lgf/include/aab/*.h"
)
set(
    sources
    lgf/operation.cpp
    lgf/dgraph.cpp
    lgf/painter.cpp
)

create_cpp_library(
    NAME lgf_lib
    #HDRS
    #${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS /EHsc
)