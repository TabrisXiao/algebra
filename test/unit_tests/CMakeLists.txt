
set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME unit_tests
    SRCS 
    ${sources}

    LINK_LIBS
    lgf_lib
    INCLUDES
    ${PROJECT_SOURCE_DIR}/test/include/
    COPTS /EHsc /W0 /std:c++17
)