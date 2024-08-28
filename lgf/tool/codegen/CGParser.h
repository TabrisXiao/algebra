
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include <set>
#include "ast/ast.h"
#include "ast/parser.h"
#include "ast/context.h"
#include "ast/lexer.h"
using namespace ast;
namespace codegen
{
    class CGParser : public ::ast::parserCore
    {
    public:
        using kind = ::ast::token::kind;
        CGParser(::ast::lexer &l) : parserCore(l) {}
        virtual ~CGParser() = default;
        std::unique_ptr<astContext> parse();

        std::unique_ptr<astContext> parse_context();

        void add_type(std::string type, astDictionary *dict)
        {
            dict->add("_type_", std::move(std::make_unique<astExpr>(loc(), type)));
        }

        std::unique_ptr<astDictionary> parse_dict();

        std::unique_ptr<astList> parse_list();

        std::unique_ptr<astList> parse_set();

        std::unique_ptr<astModule> parse_module();

        bool check_if_duplicate(std::unique_ptr<astList> &node, std::string &item)
        {
            for (auto &it : node->get_content())
            {
                if (it->get_kind() != astNode::expr)
                    continue;
                auto expr = dynamic_cast<astExpr *>(it.get())->string();
                if (expr == item)
                {
                    return true;
                }
            }
            return false;
        }
    };
}

#endif