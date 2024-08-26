set(sources
    codegen.cpp 
    CGParser.cpp
)

create_cpp_binary(
    NAME codegen
    SRCS 
    ${sources}

    LINK_LIBS
    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS ${COMPILER_OPTIONS}
)