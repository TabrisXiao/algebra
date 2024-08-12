
set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME codegen
    SRCS 
    ${sources}

    INCLUDES
    ${PROJECT_SOURCE_DIR}/lgf/include/
    COPTS ${COMPILER_OPTIONS}
)