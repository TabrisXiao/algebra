
#ifndef LGF_AST_l0lexer_H
#define LGF_AST_l0lexer_H
#include <memory>
#include <string>
#include <fstream>
#include <filesystem>
#include "utils/exception.h"
#include "utils/stream.h"
using namespace utils;
namespace lgf::ast
{

    struct location
    {
        std::shared_ptr<std::string> file; ///< filename.
        int line;                          ///< line number.
        int col;                           ///< column number.
        std::string print()
        {
            return (*file) + "(" + std::to_string(line) + ", " + std::to_string(col) + ")";
        }
    };

    class lexer
    {
    public:
        enum l0token : int
        {
            tok_eof = -1,
            tok_identifier = -2,
            tok_number = -3,
            tok_other = -4,
        };
        lexer() = default;
        lexer(lexer &l)
        {
            inherit(l);
        }
        void inherit(lexer &l)
        {
            fs = l.fs;
            lastTok = l.lastTok;
            identifierStr = l.identifierStr;
            number = l.number;
            fs = &(l.get_input_stream());
        }
        void set_input_stream(fiostream &fs)
        {
            this->fs = &fs;
            lastTok = tok_eof;
            identifierStr = "";
            number = 0;
        }
        fiostream &get_input_stream() const
        {
            return *fs;
        }
        virtual ~lexer() = default;

        static constexpr bool isalpha(unsigned ch)
        {
            return (ch | 32) - 'a' < 26;
        }

        static constexpr bool isdigit(unsigned ch) { return (ch - '0') < 10; }

        static constexpr bool isalnum(unsigned ch)
        {
            return isalpha(ch) || isdigit(ch);
        }

        static constexpr bool isgraph(unsigned ch) { return 0x20 < ch && ch < 0x7f; }

        static constexpr bool islower(unsigned ch) { return (ch - 'a') < 26; }

        static constexpr bool isupper(unsigned ch) { return (ch - 'A') < 26; }

        static constexpr bool isspace(unsigned ch)
        {
            return ch == ' ' || (ch - '\t') < 5;
        }

        char get_next_char()
        {
            return fs->get_next_char();
        }
        char get_cur_char()
        {
            return fs->get_cur_char();
        }

        virtual int get_next_token()
        {
            return get_next_l0token();
        }

        int get_next_l0token()
        {
            auto curChar = get_cur_char();
            // skip all space
            while (isspace(curChar))
                curChar = get_next_char();
            // Number: [0-9] ([0-9.])*
            if (isdigit(curChar))
            {
                std::string numStr;
                do
                {
                    numStr += curChar;
                    curChar = get_next_char();
                } while (isdigit(curChar) || curChar == '.');
                number = strtod(numStr.c_str(), nullptr);
                lastTok = tok_number;
            }
            // Identifier: [a-zA-Z][a-zA-Z0-9_]*
            else if (isalpha(curChar))
            {
                identifierStr = curChar;
                while (isalnum((curChar = l0token(get_next_char()))) || curChar == '_')
                {
                    identifierStr += curChar;
                }
                lastTok = tok_identifier;
            }
            else if (curChar == EOF)
            {
                if (fs->is_eof())
                    return tok_eof;
                else
                {
                    curChar = get_next_char();
                    return get_next_l0token();
                }
            }
            else
            {
                identifierStr = curChar;
                lastTok = tok_other;
                get_next_char();
            }
            while (isspace(curChar))
                curChar = get_next_char();
            return lastTok;
        }

        void parse(l0token tok)
        {
            get_next_l0token();
            if (tok != lastTok)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected token." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        virtual void parse(std::string str)
        {
            get_next_l0token();
            if (lastTok != tok_other && lastTok != tok_identifier)
            {
                std::cerr << get_loc().print() << ": consuming an string but got other token" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (str != identifierStr)
            {
                std::cerr << get_loc().print() << ": consuming string \"" << str << "\" but got: \"" << identifierStr << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        l0token last_tok()
        {
            return l0token(lastTok);
        }

        fiostream::location get_loc() { return fs->get_loc(); }

        std::string get_string() { return identifierStr; }
        int lastTok;
        // If the current Token is an identifier, this string contains the value.
        std::string identifierStr;
        // If the current Token is a number, this variable contains the value.
        double number;
        fiostream *fs = nullptr;
    };

    class cLikeLexer : public lexer
    {
    public:
        enum cToken : int
        {
            tok_identifier = -1,
            tok_number = -2,
            tok_eof = -100,
            tok_unknown = -999,
        };
        cLikeLexer() = default;
        cToken find_cToken(std::string str)
        {
            if (str.size() == 1)
            {
                return cToken(str[0]);
            }
            else
            {
                return cToken::tok_unknown;
            }
        }
        virtual int get_next_token() override
        {
            auto tok = get_next_l0token();
            lastTok = cToken::tok_unknown;
            if (tok == tok_identifier)
                lastTok = cToken::tok_identifier;
            else if (tok == tok_number)
                lastTok = cToken::tok_number;
            else if (tok == tok_eof)
                lastTok = cToken::tok_eof;
            else if (tok == tok_other)
            {
                lastTok = find_cToken(identifierStr);
            }
            return lastTok;
        }

        void parse(cToken tok)
        {
            get_next_token();
            if (tok != lastTok)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected token." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        void parse(std::string str) override
        {
            if (str.size() == 1)
            {
                parse(cToken(str[0]));
            }
            else
            {
                auto tok = find_cToken(str);
                get_next_token();
                if (lastTok != tok)
                {
                    std::cerr << get_loc().print() << ": consuming an unexpected token." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    };
}

#endif