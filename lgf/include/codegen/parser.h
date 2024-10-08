
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include "ast/parser.h"
#include "modules.h"
namespace lgf::codegen
{
    class codegenParserBook : public parserBook
    {
    public:
        codegenParserBook()
        {
            pmap["node"] = std::make_unique<nodeParser>();
        }
    };

    class codegenParser : public lgf::ast::parser
    {
    public:
        codegenParser()
        {
            root = std::make_unique<ast::astDictionary>();
            auto ptr = std::make_unique<ast::astList>();
            list = dynamic_cast<ast::astList *>(ptr.get());
            root->add("content", std::move(ptr));
            load_lexer<ast::cLikeLexer>();
        }
        virtual ~codegenParser() = default;
        bool parse()
        {
            // return true if error
            while (lx()->get_next_token() != ast::lexer::l0token::tok_eof)
            {
                if (lx()->last_tok() == ast::lexer::l0token::tok_identifier)
                {
                    // Parse identifier
                    std::string id = get_string();
                    if (id == "module")
                    {
                        parser_module();
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
        void parser_module()
        {
            parse_less_than();
            auto id = parse_id();
            parse_greater_than();
            auto tp = mmap.get(id);
            THROW_WHEN(tp == nullptr, "Parse error: Can't find the template: " + id);
            list->add(std::move(tp->parse(get_input_stream())));
        }
        codegenParserBook mmap;
        std::unique_ptr<ast::astDictionary> root;
        ast::astList *list = nullptr;
    };
}
#endif