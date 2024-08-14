
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
            lex.parse("<");
        }
        void parse_greater_than()
        {
            lex.parse(">");
        }
        void parse_equal()
        {
            lex.parse("=");
        }
        void parse_semicolon()
        {
            lex.parse(";");
        }
        void parse_comma()
        {
            lex.parse(",");
        }
        void parse_colon()
        {
            lex.parse(":");
        }
        void parse_left_brace()
        {
            lex.parse("{");
        }
        void parse_right_brace()
        {
            lex.parse("}");
        }
        std::string get_cur_string()
        {
            return lex.get_cur_string();
        }
        std::string parse_id()
        {
            lex.parse(l0lexer::l0token::tok_identifier);
            auto id = lex.get_cur_string();
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