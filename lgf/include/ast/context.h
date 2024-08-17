
#ifndef AST_CONTEXT_H
#define AST_CONTEXT_H
#include <map>
#include <string>
#include <iostream>
#include "utils/object.h"
#include "utils/stream.h"
#include <vector>
namespace ast
{
    class context : public utils::cursive_map<std::string, std::vector<std::string>>
    {
    public:
        context() = default;
        ~context() = default;
        void print()
        {
            utils::cgstream os;
            print_impl(*this, os);
        }
        void print_map(std::map<std::string, cursive_map<std::string, std::vector<std::string>>> &_map, cgstream &os)
        {
            for (auto &entry : _map)
            {
                os.indent() << entry.first << " { \n";
                os.incr_indent_level();
                print_impl(entry.second, os);
                os.decr_indent() << "}\n";
            }
        }
        void print_impl(utils::cursive_map<std::string, std::vector<std::string>> &table, cgstream &os)
        {
            for (const auto &entry : table.get_value())
            {
                os.indent() << entry << "\n";
            }
            print_map(table.get_map(), os);
        }
        std::string name;
    };
} // namespace ast

#endif