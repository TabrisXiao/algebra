
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include "ast/parser.h"
namespace lgf::codegen
{

    class codegenParser : public lgf::ast::parser
    {
    public:
        codegenParser() = default;
        virtual ~codegenParser() = default;
        bool parse()
        {
            // return true if error
            lx().get_next_l0token();
            while (lx().get_cur_token() != ast::l0lexer::l0token::tok_eof)
            {
                if (lx().get_cur_token() == ast::l0lexer::l0token::tok_identifier)
                {
                    // Parse identifier
                    std::string id = parser_id();
                    // Parse equal sign
                    parser_equal();
                    // Parse value
                    std::string value = parser_id();
                    // Parse semicolon
                    parser_semicolon();
                }
                else
                {
                    // Parse error
                    return true;
                }
            }
            return false;
        }
    };
}
#endif