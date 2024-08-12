
#ifndef LGF_AST_GENERATOR_H
#define LGF_AST_GENERATOR_H

#include "ast.h"
#include "parser.h"
namespace lgf::ast
{

    class generator
    {
    public:
        generator() {}
        virtual ~generator() {}
        virtual void parse() = 0;

    private:
        lgf::ast::parser l0parser;
    };

} // namespace lgf::ast

#endif