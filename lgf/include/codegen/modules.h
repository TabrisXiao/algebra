#ifndef LGF_CODEGEN_GENERATOR_H
#define LGF_CODEGEN_GENERATOR_H
#include "ast/ast.h"
#include "ast/generator.h"
#include "ast/lexer.h"
#include <map>
#include <memory>
#include "ast/stream.h"
using namespace lgf::ast;
namespace lgf::codegen
{
    class parserModule : public ast::parser
    {
    public:
        parserModule() = default;
        virtual ~parserModule() = default;
        virtual std::unique_ptr<astNode> parse(fiostream &fs) = 0;
        std::unique_ptr<astNode> node;
    };

    class parserBook
    {
    public:
        using paser_ptr = std::unique_ptr<astNode> (*)(fiostream &);
        parserBook() = default;
        ~parserBook() = default;
        parserModule *get(const std::string &id)
        {
            if (pmap.find(id) == pmap.end())
            {
                return nullptr;
            }
            return pmap[id].get();
        }
        std::map<std::string, std::unique_ptr<parserModule>> pmap;
    };

    class nodeParser : public parserModule
    {
    public:
        nodeParser() = default;
        virtual ~nodeParser() = default;
        virtual std::unique_ptr<astNode> parse(fiostream &fs) override
        {
            set_input_stream(fs);
            auto id = parse_id();
            node = std::make_unique<astModule>(id);
            parse_left_brace();
            while (get_cur_string() != "}")
            {
                auto key = parse_id();
                if (key == "input")
                {
                    parse_input();
                }
            }
            parse_right_brace();
            return std::move(node);
        }

        void parse_input()
        {
            parse_colon();
            parse_left_brace();
            while (get_cur_string() != "}")
            {
                auto type = parse_id();
                parse_colon();
                auto id = parse_id();
                func->add_arg(std::move(std::make_unique<astVar>(type, id)));
                if (get_cur_string() == ",")
                    parse_comma();
                else
                {
                    break;
                }
            }
            parse_right_brace();
        }
        std::unique_ptr<astFuncDefine> func;
    };
}

#endif