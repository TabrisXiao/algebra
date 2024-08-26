
#ifndef AOC_EXCEPTION_H
#define AOC_EXCEPTION_H
#include <exception>
#include <iostream>
#include <string>
#define CHECK_VALUE(var, val, msg)                                                                                                                                          \
    if ((var) != (val))                                                                                                                                                     \
    {                                                                                                                                                                       \
        std::cerr << "Runtime Error: " __FILE__ ":" << std::to_string(__LINE__) << " : Expecting \"" #var "\" to be " << val << ", but got " << var << ". " << msg << "\n"; \
        std::exit(EXIT_FAILURE);                                                                                                                                            \
    }
#define CHECK_CONDITION(condition, msg)                                                                                                        \
    if (!(condition))                                                                                                                          \
    {                                                                                                                                          \
        std::cerr << "Runtime Error: " __FILE__ ":" << std::to_string(__LINE__) << " : Expecting " #condition ", but failed. " << msg << "\n"; \
        std::exit(EXIT_FAILURE);                                                                                                               \
    }
#define THROW_WHEN(condition, msg)                                                                      \
    if (condition)                                                                                      \
    {                                                                                                   \
        std::cerr << "Runtime Error: " __FILE__ ":" << std::to_string(__LINE__) << ": " << msg << "\n"; \
        std::exit(EXIT_FAILURE);                                                                        \
    }

#define THROW(msg)                                                                                  \
    std::cerr << "Runtime Error: " __FILE__ ":" << std::to_string(__LINE__) << ": " << msg << "\n"; \
    std::exit(EXIT_FAILURE);

#define WARNING(msg) \
    std::cerr << "Runtime Warning: " __FILE__ ":" << std::to_string(__LINE__) << ": " << msg << "\n";

#endif