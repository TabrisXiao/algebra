set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME unit_tests
    SRCS 
    ${sources}

    LINK_LIBS
    logic_graph_lib
    COPTS /EHsc /W0 /std:c++17
)