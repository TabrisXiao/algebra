file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/utils/*.h"
)

set(
    sources
    logger.cc
)

create_cpp_library(
    NAME utils_lib
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/utils
    SRCS
    ${sources}
    COPTS ${COMPILER_OPTIONS}
)