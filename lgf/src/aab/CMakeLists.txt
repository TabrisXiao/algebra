file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/aab/*.h"
)
set(
    sources
    passes.cpp
)

create_cpp_library(
    NAME lgf_aab_lib
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