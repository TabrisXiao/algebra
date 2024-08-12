
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include "ast/parser.h"
#include "generator.h"
namespace lgf::codegen
{

    class codegenParser : public lgf::ast::parser
    {
    public:
        codegenParser()
        {
            root = std::make_unique<ast::astModule>("");
        }
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
                    std::string id = parse_id();
                    if (id == "template")
                    {
                        parser_template();
                    }
                }
                else
                {
                    // Parse error
                    return true;
                }
            }
            return false;
        }
        void parser_template()
        {
            parse_less_than();
            auto id = parse_id();
            parse_greater_than();
            auto tp = tmap.get(id)->get();
            THROW_WHEN(tp == nullptr, "Parse error: Can't find the template: " + id);
            root->add_node(std::move(tp->parse(get_input_stream())));
        }
        generatorMap tmap;
        std::unique_ptr<ast::astModule> root;
    };
}
#endif