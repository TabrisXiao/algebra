
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
        enum kind_t
        {
            unknown = 0,
            node = 1,
            attr = 2,
            desc = 3,
            context = 4,
            symbol = 5,
            region = 6 // like dict, list or set
        };
        symbolInfo() = default;
        symbolInfo(size_t c) : stype(c) {}
        std::string print()
        {
            switch (stype)
            {
            case node:
                return "node";
            case attr:
                return "attr";
            case desc:
                return "desc";
            case context:
                return "context";
            case symbol:
                return "symbol";
            case region:
                return "region";
            default:
                return "unknown";
            }
        }
        size_t stype = 0;
    };

    class CGContext
    {
    public:
        class CGCGuard
        {
        public:
            CGCGuard() = default;
            CGCGuard(CGContext *ctx) : _ctx_(ctx)
            {
                _ctx_->goto_root_table();
            }
            CGCGuard(CGContext *ctx, std::string key) : _ctx_(ctx)
            {
                _ctx_->enter(key);
            }
            CGCGuard(CGContext *ctx, std::string key, symbolInfo info) : _ctx_(ctx)
            {
                _ctx_->create_and_enter(key, info);
            }
            ~CGCGuard()
            {
                if (_ctx_)
                    _ctx_->back_to_previous_table();
            }
            CGContext *_ctx_ = nullptr;
        };
        CGContext() : _table_(symbolInfo(symbolInfo::kind_t::context))
        {
        }
        ~CGContext() = default;

        void reset()
        {
            _stack_ = {&_table_};
        }

        void print()
        {
            aoc::stringBufferProducer os;
            print_table("_root_", _table_, os);
            std::cout << os.str();
        }
        void print_map(std::map<std::string, aoc::cursive_map<std::string, symbolInfo>> &_map, aoc::stringBufferProducer &os)
        {
            os.indent() << "{\n";
            os.indent_level_up();
            for (auto &entry : _map)
            {
                print_table(entry.first, entry.second, os);
            }
            os.decr_indent() << "}\n";
        }
        void print_table(std::string key, aoc::cursive_map<std::string, symbolInfo> &table, aoc::stringBufferProducer &os)
        {
            os.indent() << table.get_value().print() << " " << key << "\n";
            if (table.has_map())
                print_map(table.get_map(), os);
        }
        symbolInfo *get_info(std::string name)
        {
            for (auto &entry : _stack_)
            {
                auto item = entry->get(name);
                if (item)
                {
                    return &(entry->get(name)->get_value());
                }
            }
            return nullptr;
        }

        aoc::cursive_map<std::string, symbolInfo> *get_current_table()
        {
            return _stack_.back();
        }

        void create_table(std::string key, symbolInfo &info)
        {
            get_current_table()->add(key, info);
        }

        void create_and_enter(std::string key, symbolInfo &info)
        {
            auto ptr = get_current_table()->get(key);
            if (ptr != nullptr)
            {
                _stack_.push_back(ptr);
                return;
            }
            create_table(key, info);
            auto nptr = get_current_table()->get(key);
            _stack_.push_back(nptr);
        }

        void enter(std::string key)
        {
            if (!get_current_table()->has(key))
            {
                std::runtime_error("enter: key doesn't exists!");
            }
            auto ptr = get_current_table()->get(key);
            _stack_.push_back(ptr);
        }

        void back_to_previous_table()
        {
            if (_stack_.size() == 1)
            {
                std::runtime_error("back_to_previous_table: already at the root table!");
            }
            _stack_.pop_back();
        }

        void create_context_table(std::string key)
        {
            symbolInfo info(symbolInfo::context);
            create_and_enter(key, info);
        }

        void create_node_table(std::string key)
        {
            symbolInfo info(symbolInfo::node);
            create_and_enter(key, info);
        }
        void create_attr_table(std::string key)
        {
            symbolInfo info(symbolInfo::attr);
            create_and_enter(key, info);
        }
        void create_desc_table(std::string key)
        {
            symbolInfo info(symbolInfo::desc);
            create_and_enter(key, info);
        }

        void create_attr(std::string key)
        {
            symbolInfo info(symbolInfo::attr);
            create_table(key, info);
        }
        void create_symbol(std::string key)
        {
            symbolInfo info(symbolInfo::symbol);
            create_table(key, info);
        }

        void goto_root_table()
        {
            _stack_.push_back(&_table_);
        }

        std::string name;
        aoc::cursive_map<std::string, symbolInfo> _table_;
        std::vector<aoc::cursive_map<std::string, symbolInfo> *> _stack_ = {&_table_};
    };
} // namespace codegen

#endif