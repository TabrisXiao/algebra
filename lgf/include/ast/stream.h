#ifndef LGF_AST_STREAM_H
#define LGF_AST_STREAM_H
#include <filesystem>
#include <fstream>
#include <memory>
#include "lgf/exception.h"
namespace lgf::ast
{
    class fiostream
    {
    public:
        struct location
        {
            std::shared_ptr<std::filesystem::path> path; ///< filename.
            uint line = 0, col = 0;
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

        char get_next_char()
        {
            if (buffer.empty() || cur == buffer.end())
            {
                get_next_line();
            }
            if (buffer.empty())
            {
                return EOF;
            }
            loc.col++;
            return *cur++;
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
} // namespace lgf::ast

#endif