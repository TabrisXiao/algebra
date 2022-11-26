
create_cpp_binary(
    NAME test_compiler
    SRCS main.cpp
    DEPS
    ${COMPILER_MODULE}
    COPTS /EHsc /W0 /std:c++17
)