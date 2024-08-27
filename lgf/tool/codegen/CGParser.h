
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include <set>
#include "ast/ast.h"
#include "ast/parser.h"
#include "ast/context.h"
#include "ast/lexer.h"

class CGParser : public ast::parserCore
{
public:
    using kind = ast::token::kind;
    CGParser(ast::lexer &l) : parserCore(l) {}
    virtual ~CGParser() = default;
    std::unique_ptr<ast::astDictionary> parse();

    std::unique_ptr<ast::astDictionary> parse_context();

    void add_type(std::string type, ast::astDictionary *dict)
    {
        dict->add("_type_", std::move(std::make_unique<ast::astExpr>(loc(), type)));
    }

    std::unique_ptr<ast::astDictionary> parse_dict();

    std::unique_ptr<ast::astList> parse_list();

    std::unique_ptr<ast::astList> parse_set();

    std::unique_ptr<ast::astModule> parse_module();

    bool check_if_duplicate(std::unique_ptr<ast::astList> &node, std::string &item)
    {
        for (auto &it : node->get_content())
        {
            if (it->get_kind() != ast::astType::expr)
                continue;
            auto expr = dynamic_cast<ast::astExpr *>(it.get())->get_expr();
            if (expr == item)
            {
                return true;
            }
        }
        return false;
    }
};

#endif