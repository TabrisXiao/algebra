
file(GLOB_RECURSE headers
    "${PROJECT_SOURCE_DIR}/lgf/include/compiler/*.h"
)
set(
    sources
    lexer.cpp
)

set(exe_source
    main.cpp 
)

create_cpp_library(
    NAME lgf_compiler_lib
    #HDRS
    #${headers}
    SRCS
    ${sources}
    LINK_LIBS
    lgf_lib
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS ${COMPILER_OPTIONS}
)

create_cpp_binary(
    NAME compiler
    SRCS 
    ${exe_source}

    LINK_LIBS
    lgf_lib
    lgf_compiler_lib
    COPTS ${COMPILER_OPTIONS}
)