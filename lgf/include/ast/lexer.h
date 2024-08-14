
#ifndef LGF_AST_l0lexer_H
#define LGF_AST_l0lexer_H
#include <memory>
#include <string>
#include <fstream>
#include <filesystem>
#include "lgf/exception.h"
#include "stream.h"
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

    class l0lexer
    {
    public:
        enum l0token : int
        {
            tok_eof = -1,
            tok_identifier = 1,
            tok_number = 2,
            tok_other = 3,
            tok_unknown = 999
        };
        l0lexer() = default;
        l0lexer(l0lexer &l)
        {
            inherit(l);
        }
        void inherit(l0lexer &l)
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
        ~l0lexer() = default;

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

        l0token get_next_l0token()
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
                return lastTok;
            }
            // Identifier: [a-zA-Z][a-zA-Z0-9_]*
            if (isalpha(curChar))
            {
                identifierStr = curChar;
                while (isalnum((curChar = l0token(get_next_char()))) || curChar == '_')
                {
                    identifierStr += curChar;
                }
                lastTok = tok_identifier;
                return lastTok;
            }
            if (curChar == EOF)
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
                while (!isspace(curChar = get_next_char()) && !isalnum(curChar) && curChar != EOF)
                {
                    identifierStr += curChar;
                }
                lastTok = tok_other;
                return lastTok;
            }
            lastTok = tok_unknown;
            return lastTok;
        }

        void parse(l0token tok)
        {
            get_next_l0token();
            if (tok != lastTok)
            {
                auto symbol = convert_current_token2string();
                std::cerr << get_loc().print() << ": consuming an unexpected token \"" << convert_current_token2string() << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        void parse(std::string str)
        {
            get_next_l0token();
            if (lastTok != tok_other)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected token \"" << convert_current_token2string() << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (str != identifierStr)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected inputs \"" << identifierStr << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        l0token get_cur_token()
        {
            return lastTok;
        }

        std::string convert_current_token2string()
        {
            if (lastTok >= 0)
                return std::string(1, static_cast<char>(lastTok));
            else
                return identifierStr;
        }
        fiostream::location get_loc() { return fs->get_loc(); }

        std::string get_cur_string() { return identifierStr; }
        l0token lastTok;
        // If the current Token is an identifier, this string contains the value.
        std::string identifierStr;
        // If the current Token is a number, this variable contains the value.
        double number;
        lgf::ast::fiostream *fs = nullptr;
    };
}

#endif