

#ifndef CODEGEN_STREAM_H_
#define CODEGEN_STREAM_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "lgf/exception.h"

namespace lgf::codegen
{

    class cgstream
    {
    public:
        cgstream() = default;
        // makes the singleton to be unassignable / non-clonable
        // stream(stream &) = delete;
        ~cgstream() = default;
        // void operator=(const stream &) = delete;
        void live_stream_to_console()
        {
            outputf.close();
            os = &std::cout;
        }
        void live_stream_to_file(const std::string &filename)
        {
            outputf.open(filename);
            os = &outputf;
        }
        void stream_to_buffer()
        {
            outputf.close();
            os = &ss;
        }
        std::string get_buffer_content()
        {
            return ss.str();
        }
        void dump_to_console()
        {
            std::cout << get_buffer_content() << std::endl;
        }
        void dump_to_file(const std::string &filename)
        {
            live_stream_to_file(filename);
            *os << get_buffer_content();
            outputf.close();
        }
        template <typename T>
        cgstream &operator<<(const T &data)
        {
            *os << data;
            return *this;
        }
        void print_indent()
        {
            for (int i = 0; i < curIndentLevel; i++)
                *os << "  ";
        }
        void incr_indent_level(int n = 1) { curIndentLevel += n; }
        void decr_indent_level(int n = 1)
        {
            curIndentLevel -= n;
            if (curIndentLevel < 0)
                curIndentLevel = 0;
        }

    protected:
        std::ostream *os = &std::cout;
        std::stringstream ss;
        std::ofstream outputf;
        int curIndentLevel = 0;
    };

    class indentGuard
    {
    public:
        indentGuard() = delete;
        indentGuard(cgstream &sg)
        {
            st = &sg;
            st->incr_indent_level();
        }
        ~indentGuard()
        {
            st->decr_indent_level();
        }
        cgstream *st = nullptr;
    };
} // namespace lgf::codegen

#endif