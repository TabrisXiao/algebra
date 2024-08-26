
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
    std::unique_ptr<ast::astDictionary> &&parse();

    std::unique_ptr<ast::astDictionary> &&parse_context();

    std::unique_ptr<ast::astModule> &&parser_module()
    {
    }

    void add_type(std::string type, ast::astDictionary *dict)
    {
        dict->add("_type_", std::move(std::make_unique<ast::astExpr>(loc(), type)));
    }

    std::unique_ptr<ast::astDictionary> parse_dict();

    std::unique_ptr<ast::astList> &&parse_list();

    std::unique_ptr<ast::astList> &&parse_set();

    std::unique_ptr<ast::astModule> &&parse_module();
};

#endif