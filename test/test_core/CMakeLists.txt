set(sources
    main.cpp 
    tests/test_aog.h
)
create_cpp_binary(
    NAME test_core
    SRCS 
    ${sources}

    LINK_LIBS
    logic_abstract_lib
    COPTS /EHsc /W0 /std:c++17
)