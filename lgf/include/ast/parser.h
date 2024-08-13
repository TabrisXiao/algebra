
#ifndef LGF_AST_PARSER_H
#define LGF_AST_PARSER_H
#include "lexer.h"
namespace lgf::ast
{
    class parser
    {
    public:
        parser() = default;
        virtual ~parser() = default;
        parser(const parser &p_)
        {
            lex.set_input_stream(p_.get_input_stream());
        }
        void set_input_stream(fiostream &fs)
        {
            lex.set_input_stream(fs);
        }
        void parse_less_than()
        {
            lex.consume_special("<");
        }
        void parse_greater_than()
        {
            lex.consume_special(">");
        }
        void parse_equal()
        {
            lex.consume_special("=");
        }
        void parse_semicolon()
        {
            lex.consume_special(";");
        }
        void parse_comma()
        {
            lex.consume_special(",");
        }
        void parse_colon()
        {
            lex.consume_special(":");
        }
        void parse_left_brace()
        {
            lex.consume_special("{");
        }
        void parse_right_brace()
        {
            lex.consume_special("}");
        }
        std::string get_cur_string()
        {
            return lex.get_cur_string();
        }
        std::string parse_id()
        {
            auto id = lex.get_cur_string();
            lex.consume(l0lexer::l0token::tok_identifier);
            return id;
        }
        fiostream &get_input_stream() const
        {
            return lex.get_input_stream();
        }
        l0lexer &lx()
        {
            return lex;
        }

    private:
        l0lexer lex;
    };
} // namespace lgf::ast

#endif