#ifndef UTILS_STREAM_H
#define UTILS_STREAM_H
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include "exception.h"
namespace utils
{
    class fiostream
    {
    public:
        struct location
        {
            std::shared_ptr<std::filesystem::path> path; ///< filename.
            unsigned int line = 0, col = 0;
            std::string print()
            {
                return path->string() + "(" + std::to_string(line) + ", " + std::to_string(col) + ")";
            }
        };
        fiostream() = default;
        virtual ~fiostream() = default;
        void load_file(std::string filename)
        {
            loc.path = std::make_shared<std::filesystem::path>(std::filesystem::absolute(filename));
            file.open(*loc.path.get());
            THROW_WHEN(!file.is_open(), "fiostream error: Can't open the file: " + loc.path->string());
            get_next_line();
        }
        std::filesystem::path get_file_path()
        {
            return *(loc.path.get());
        }
        virtual void get_next_line()
        {
            if (std::getline(file, buffer))
            {
                cur = buffer.begin();

                loc.line++;
                loc.col = 1;
            }
            else
            {
                buffer.clear();
            }
        }
        location get_loc()
        {
            return loc;
        }

        char get_cur_char()
        {
            if (cur == buffer.end())
            {
                return EOF;
            }
            return *cur;
        }

        char get_next_char()
        {
            if (buffer.empty() || (cur + 1) == buffer.end())
            {
                get_next_line();
                if (buffer.empty())
                {
                    return EOF;
                }
                return *cur;
            }
            loc.col++;
            cur = cur + 1;
            return *(cur);
        }
        bool is_eof()
        {
            return file.eof();
        }

    private:
        std::string buffer;
        std::string::iterator cur;
        std::ifstream file;
        location loc;
    };

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
        cgstream &indent()
        {
            for (int i = 0; i < curIndentLevel; i++)
                *os << "    ";
            return *this;
        }
        cgstream &incr_indent()
        {
            incr_indent_level();
            return indent();
        }
        cgstream &decr_indent()
        {
            decr_indent_level();
            return indent();
        }
        void incr_indent_level(int n = 1) { curIndentLevel += n; }
        void decr_indent_level(int n = 1)
        {
            curIndentLevel -= n;
            if (curIndentLevel < 0)
                curIndentLevel = 0;
        }
        std::ostream *os = &std::cout;

    protected:
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
} // namespace utils

#endif