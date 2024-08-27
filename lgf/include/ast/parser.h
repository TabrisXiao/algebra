
#ifndef AST_PARSER_H
#define AST_PARSER_H
#include "lexer.h"
#include "aoc/convention.h"
namespace ast
{
    class parserCore
    {
    public:
        parserCore() = delete;
        parserCore(lexer &lx_) : lx(lx_) {}
        virtual ~parserCore() = default;

        void emit_error(const std::string &msg)
        {
            // std::cerr << lx.loc().print() << msg << std::endl;
            throw std::runtime_error(lx.loc().print() + msg);
        }
        void emit_error_if(bool condition, const std::string &msg)
        {
            if (condition)
                emit_error(msg);
        }

        void consume(const token::kind k)
        {
            if (curTok.is(k))
            {
                curTok = lx.lex_token();
                return;
            }
            emit_error("consumed an unexpected token.");
        }

        void consume()
        {
            emit_error_if(curTok.is(token::tok_eof), "parser reached to EOF.");
            curTok = lx.lex_token();
        }

        aoc::logicResult try_consume(const token::kind k)
        {
            if (curTok.is(k))
            {
                consume(k);
                return aoc::logicResult::success();
            }
            return aoc::logicResult::fail();
        }

        void parse_less_than()
        {
            consume(token::kind('<'));
        }
        void parse_greater_than()
        {
            consume(token::kind('>'));
        }
        void parse_equal()
        {
            consume(token::kind('='));
        }
        void parse_semicolon()
        {
            consume(token::kind(';'));
        }
        void parse_comma()
        {
            consume(token::kind(','));
        }
        void parse_colon()
        {
            consume(token::kind(':'));
        }
        void parse_dot()
        {
            consume(token::kind('.'));
        }
        void parse_left_bracket()
        {
            consume(token::kind('['));
        }
        void parse_right_bracket()
        {
            consume(token::kind(']'));
        }
        void parse_left_brace()
        {
            consume(token::kind('{'));
        }
        void parse_right_brace()
        {
            consume(token::kind('}'));
        }
        void parse_scope()
        {
            consume(token::tok_scope);
        }
        void parse_div()
        {
            consume(token::kind('/'));
        }
        void parse_mul()
        {
            consume(token::kind('*'));
        }
        void parse_mod()
        {
            consume(token::kind('%'));
        }
        void parse_add()
        {
            consume(token::kind('+'));
        }
        void parse_sub()
        {
            consume(token::kind('-'));
        }
        void parse_and()
        {
            consume(token::kind('&'));
        }
        void parse_or()
        {
            consume(token::kind('|'));
        }
        void parse_xor()
        {
            consume(token::kind('^'));
        }
        void parse_apostrophe()
        {
            consume(token::kind('\''));
        }

        std::string parse_id()
        {
            emit_error_if(!curTok.is(token::tok_identifier), "Expecting an identifier.");
            auto ref = curTok.get_string();
            consume(token::tok_identifier);
            return ref.data();
        }
        lexer &get_lexer()
        {
            return lx;
        }

        charLocation loc()
        {
            return lx.loc();
        }
        std::string parse_number()
        {
            emit_error_if(!curTok.is(token::tok_integer) && !curTok.is(token::tok_float), "Expecting a number.");
            auto ref = curTok.get_string();
            try_consume(token::tok_integer);
            try_consume(token::tok_float);
            return ref.data();
        }

        token cur_tok()
        {
            return curTok;
        }

    private:
        lexer &lx;
        token curTok;
    };
} // namespace compiler

#endif