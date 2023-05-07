set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME integrate_tests
    SRCS 
    ${sources}

    LINK_LIBS
    lgf_lib
    lgf_aab_lib
    COPTS /EHsc /W0 /std:c++17
    INCLUDES
    ${PROJECT_SOURCE_DIR}/test/include/
)