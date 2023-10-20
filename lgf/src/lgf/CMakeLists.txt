file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/lgf/*.h"
)
set(
    sources
    operation.cpp   
    pass.cpp
    libs/AAB/ops.cpp
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