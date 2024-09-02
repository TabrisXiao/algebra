#ifndef AST_LEXER_H
#define AST_LEXER_H

#include <memory>
#include <string>
#include <fstream>
#include <filesystem>
#include "aoc/exception.h"
#include "aoc/stream.h"
#include "aoc/object.h"

namespace ast
{
    class charLocation
    {
    public:
        charLocation() = delete;
        charLocation(unsigned int l, unsigned int c, std::shared_ptr<std::string> &p) : line(l), col(c), path(p) {}
        ~charLocation() = default;
        std::shared_ptr<std::string> path; ///< filename.
        unsigned int line = 1, col = 1;
        std::string print()
        {
            return *(path.get()) + "(" + std::to_string(line) + ", " + std::to_string(col) + ")";
        }
    };

    class token
    {
    public:
        enum kind : int
        {
            tok_eof = -1,
            tok_identifier = -2,
            tok_integer = -3,
            tok_float = -4,
            tok_scope = -5,
        };
        token() = default;
        token(kind k, aoc::stringRef &str) : _k(k), ref(str) {}
        ~token() = default;

        kind get_kind() const { return _k; }
        aoc::stringRef get_ref() { return ref; }
        std::string get_string() { return ref.data(); }
        bool is(kind k) const { return _k == k; }
        bool is_any(kind k1, kind k2) const { return is(k1) || is(k2); }

        /// Return true if this token is one of the specified kinds.
        template <typename... T>
        bool is_any(kind k1, kind k2, kind k3, T... others) const
        {
            if (is(k1))
                return true;
            return is_any(k2, k3, others...);
        }

    private:
        kind _k;
        aoc::stringRef ref;
    };

    class lexer
    {
    public:
        lexer() = default;
        lexer(std::string loc_p, aoc::stringBuf &&buf)
        {
            load_stringBuf(loc_p, std::move(buf));
        }
        void load_stringBuf(std::string loc_p, aoc::stringBuf &buf)
        {
            buffer.release();
            locPath = std::make_shared<std::string>(loc_p);
            curPtr = buf.begin();
            lastLine = curPtr;
            buffer = std::move(buf);
            line = 1;
        }
        void resetPointer(const char *ptr)
        {
            curPtr = ptr;
        }

        void emit_error(const std::string &msg)
        {
            throw std::runtime_error(loc().print() + msg);
        }
        void emit_error_if(bool condition, const std::string &msg)
        {
            if (condition)
                emit_error(msg);
        }

        token lex_token()
        {
            while (true)
            {
                const char *start = curPtr;
                switch (*curPtr++)
                {

                case ' ':
                    continue;
                case '\n':
                case '\r':
                case '\t':
                    line++;
                    lastLine = curPtr;
                    continue;
                case 0:
                    // file end
                    if (curPtr - 1 == buffer.end())
                        return token(token::tok_eof, aoc::stringRef(curPtr - 1, 1));
                    continue;
                case ':':
                    if (*curPtr == ':')
                        return token(token::tok_scope, aoc::stringRef(curPtr, 2));
                    return token(token::kind(':'), aoc::stringRef(curPtr, 1));
                case '/':
                    return token(token::kind('/'), aoc::stringRef(curPtr, 1));
                case '}':
                    return token(token::kind('}'), aoc::stringRef(curPtr, 1));
                case '{':
                    return token(token::kind('{'), aoc::stringRef(curPtr, 1));
                case '[':
                    return token(token::kind('['), aoc::stringRef(curPtr, 1));
                case ']':
                    return token(token::kind(']'), aoc::stringRef(curPtr, 1));
                case ',':
                    return token(token::kind(','), aoc::stringRef(curPtr, 1));
                case ';':
                    return token(token::kind(';'), aoc::stringRef(curPtr, 1));
                case '.':
                    return token(token::kind('.'), aoc::stringRef(curPtr, 1));
                case '+':
                    return token(token::kind('+'), aoc::stringRef(curPtr, 1));
                case '-':
                    return token(token::kind('-'), aoc::stringRef(curPtr, 1));
                case '*':
                    return token(token::kind('*'), aoc::stringRef(curPtr, 1));
                case '%':
                    return token(token::kind('%'), aoc::stringRef(curPtr, 1));
                case '&':
                    return token(token::kind('&'), aoc::stringRef(curPtr, 1));
                case '|':
                    return token(token::kind('|'), aoc::stringRef(curPtr, 1));
                case '^':
                    return token(token::kind('^'), aoc::stringRef(curPtr, 1));
                case '<':
                    return token(token::kind('<'), aoc::stringRef(curPtr, 1));
                case '>':
                    return token(token::kind('>'), aoc::stringRef(curPtr, 1));
                case '=':
                    return token(token::kind('='), aoc::stringRef(curPtr, 1));
                case '!':
                    return token(token::kind('!'), aoc::stringRef(curPtr, 1));
                case '?':
                    return token(token::kind('?'), aoc::stringRef(curPtr, 1));
                case '~':
                    return token(token::kind('~'), aoc::stringRef(curPtr, 1));
                case '#':
                    return token(token::kind('#'), aoc::stringRef(curPtr, 1));
                case '@':
                    return token(token::kind('@'), aoc::stringRef(curPtr, 1));
                case '$':
                    return token(token::kind('$'), aoc::stringRef(curPtr, 1));
                case '\\':
                    return token(token::kind('\\'), aoc::stringRef(curPtr, 1));
                case '\'':
                    return token(token::kind('\''), aoc::stringRef(curPtr, 1));
                case '"':
                    return token(token::kind('"'), aoc::stringRef(curPtr, 1));
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    return lex_number();
                default:
                    if (isalpha(*start) || *start == '_')
                        return lex_identifier();
                    char c = *start;
                    THROW("Unexpected character: " + std::string(1, c));
                }
            }
        }

        token lex_number()
        {
            THROW_WHEN(isdigit(*curPtr), "Char \"" + std::string(1, *curPtr) + "\" is not a digit.");
            const char *start = curPtr;
            while (isdigit(*curPtr))
            {
                curPtr++;
            }
            if (*curPtr != '.')
                return token(token::tok_integer, aoc::stringRef(start, curPtr - start));
            ++curPtr;
            while (isdigit(*curPtr))
            {
                curPtr++;
            }
            if (*curPtr == 'e' || *curPtr == 'E')
            {
                if (isdigit(static_cast<unsigned char>(curPtr[1])) ||
                    ((curPtr[1] == '-' || curPtr[1] == '+') &&
                     isdigit(static_cast<unsigned char>(curPtr[2]))))
                {
                    curPtr += 2;
                    while (isdigit(*curPtr))
                        ++curPtr;
                }
            }
            return token(token::tok_float, aoc::stringRef(start, curPtr - start));
        }

        token lex_identifier()
        {
            const char *start = curPtr - 1;
            emit_error_if(!isalpha(*start) && *start != '_', "Char \"" + std::string(1, *curPtr) + "\" is not a letter.");
            while (isalnum(*curPtr) || *curPtr == '_')
            {
                curPtr++;
            }
            return token(token::tok_identifier, aoc::stringRef(start, curPtr - start));
        }

        charLocation loc()
        {
            return charLocation(line, size_t(curPtr) - size_t(lastLine), locPath);
        }

        void skip_line()
        {
            while (*curPtr != '\n' && *curPtr != '\r')
            {
                curPtr++;
            }
            lastLine = curPtr;
            line++;
        }

    private:
        unsigned int line = 1;
        aoc::stringBuf buffer;
        std::shared_ptr<std::string> locPath;
        const char *curPtr, *lastLine;
    };
}

#endif