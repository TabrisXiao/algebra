
#ifndef AST_CONTEXT_H
#define AST_CONTEXT_H
#include <map>
#include <string>
#include <iostream>
#include "aoc/object.h"
#include "aoc/stream.h"
#include <vector>
namespace ast
{
    using scode = size_t;
    class context : public aoc::cursive_map<std::string, scode>
    {
    public:
        context() = default;
        ~context() = default;
        void print()
        {
            aoc::cgstream os;
            os << "context: " << name << " {\n";
            os.incr_indent_level();
            print_impl(*this, os);
            os.decr_indent() << "}\n";
        }
        void print_map(std::map<std::string, cursive_map<std::string, scode>> &_map, aoc::cgstream &os)
        {
            for (auto &entry : _map)
            {
                os.indent() << entry.first << " { \n";
                os.incr_indent_level();
                print_impl(entry.second, os);
                os.decr_indent() << "}\n";
            }
        }
        void print_impl(aoc::cursive_map<std::string, scode> &table, aoc::cgstream &os)
        {
            print_map(table.get_map(), os);
        }
        std::string name;
    };
} // namespace ast

#endif