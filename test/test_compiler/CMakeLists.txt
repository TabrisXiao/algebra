
create_cpp_binary(
    NAME test_compiler
    SRCS main.cpp
    LINK_LIBS
    ${algebra_dialect_library}
    MLIRIR
    COPTS /EHsc /W0 /std:c++17
)