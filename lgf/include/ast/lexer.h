
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
            tok_special = 3,
        };
        l0lexer() = default;
        l0lexer(l0lexer &l)
        {
            fs = l.fs;
            lastChar = l.lastChar;
            curTok = l.curTok;
            identifierStr = l.identifierStr;
            number = l.number;
        }
        void set_input_stream(fiostream &fs)
        {
            this->fs = &fs;
            lastChar = ' ';
            curTok = tok_eof;
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

        l0token get_next_l0token()
        {
            // skip all space
            while (isspace(lastChar))
                lastChar = get_next_char();
            // Number: [0-9] ([0-9.])*
            if (isdigit(lastChar))
            {
                std::string numStr;
                do
                {
                    numStr += lastChar;
                    lastChar = get_next_char();
                } while (isdigit(lastChar) || lastChar == '.');
                number = strtod(numStr.c_str(), nullptr);
                return tok_number;
            }
            // Identifier: [a-zA-Z][a-zA-Z0-9_]*
            if (isalpha(lastChar))
            {
                identifierStr = lastChar;
                while (isalnum((lastChar = l0token(get_next_char()))) || lastChar == '_')
                    identifierStr += lastChar;
                return l0token::tok_identifier;
            }
            if (lastChar == EOF)
            {
                if (fs->is_eof())
                    return tok_eof;
                else
                {
                    lastChar = get_next_char();
                    return get_next_l0token();
                }
            }
            else
            {
                identifierStr = "";
                while (!isspace(lastChar) && lastChar != EOF)
                {
                    identifierStr += lastChar;
                    lastChar = get_next_char();
                }
            }
            return l0token::tok_special;
        }

        void consume(l0token tok)
        {
            if (tok != curTok)
            {
                auto symbol = convert_current_token2string();
                std::cerr << get_loc().print() << ": consuming an unexpected token \"" << convert_current_token2string() << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            get_next_l0token();
        }
        void consume_special(std::string str)
        {
            if (curTok != tok_special)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected token \"" << convert_current_token2string() << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (str != identifierStr)
            {
                std::cerr << get_loc().print() << ": consuming an unexpected inputs \"" << identifierStr << "\"." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            get_next_l0token();
        }
        l0token get_cur_token()
        {
            return curTok;
        }

        std::string convert_current_token2string()
        {
            if (curTok >= 0)
                return std::string(1, static_cast<char>(curTok));
            else
                return identifierStr;
        }
        fiostream::location get_loc() { return fs->get_loc(); }

        std::string get_cur_string() { return identifierStr; }
        l0token curTok;
        // If the current Token is an identifier, this string contains the value.
        std::string identifierStr;
        // If the current Token is a number, this variable contains the value.
        double number;
        char lastChar = ' ';
        lgf::ast::fiostream *fs = nullptr;
    };
}

#endif