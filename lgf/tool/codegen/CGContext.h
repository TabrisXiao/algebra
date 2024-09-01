
#ifndef CODEGEN_CONTEXT_H
#define CODEGEN_CONTEXT_H
#include <map>
#include <string>
#include <iostream>
#include "aoc/object.h"
#include "aoc/stream.h"
#include <vector>
namespace codegen
{
    class symbolInfo
    {
    public:
        enum
        {
            unknown = 0,
            node = 1,
            attr = 2,
            context = 3,
        };
        symbolInfo() = default;
        symbolInfo(size_t c) : stype(c) {}
        void print(aoc::stringBufferProducer &os)
        {
            switch (stype)
            {
            case node:
                os.indent() << "node";
            case attr:
                os.indent() << "attr";
            default:
                os.indent() << "unknown";
            }
        }
        size_t stype = 0;
    };

    class CGContext
    {
    public:
        CGContext() = default;
        ~CGContext() = default;
        void print()
        {
            aoc::stringBufferProducer os;
            os << "context: " << name << "\n";
            os.incr_indent() << "{\n";
            print_impl(_table_, os);
            os.decr_indent() << "}\n";
            std::cout << os.str() << std::endl;
        }
        void print_map(std::map<std::string, aoc::cursive_map<std::string, symbolInfo>> &_map, aoc::stringBufferProducer &os)
        {
            os << " : {\n";
            os.incr_indent();
            for (auto &entry : _map)
            {
                os.indent() << entry.first << "\n";
                os.incr_indent() << "{\n";
                print_impl(entry.second, os);
            }
            os.decr_indent() << "}\n";
        }
        void print_impl(aoc::cursive_map<std::string, symbolInfo> &table, aoc::stringBufferProducer &os)
        {
            table.get_value().print(os);
            if (table.has_map())
                print_map(table.get_map(), os);
        }
        symbolInfo *get_info(std::string name)
        {
            for (auto &entry : _stack_)
            {
                if (entry.find(name) != entry.end())
                {
                    return &entry[name].get_value();
                }
            }
            return nullptr;
        }

        std::string name;
        aoc::cursive_map<std::string, symbolInfo> _table_;
        std::vector<std::map<std::string, aoc::cursive_map<std::string, symbolInfo>>> _stack_;
    };
} // namespace ast

#endif