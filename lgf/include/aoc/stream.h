#ifndef AOC_STREAM_H
#define AOC_STREAM_H
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include "exception.h"
#include <map>
#include <functional>
namespace aoc
{

    class stringBuf : public std::string
    {
    public:
        stringBuf() = default;
        ~stringBuf() = default;
        stringBuf(const char *resource, size_t size)
        {
            source = std::make_shared<std::string>(resource);
            resize(size);
        }
        std::shared_ptr<std::string> &get_source_path() { return source; }
        const char *get_buffer_end_pointer()
        {
            return &(*this)[size()];
        }

    private:
        std::shared_ptr<std::string>
            source;
    };

    class bufferManager
    {
    public:
        bufferManager() = default;
        size_t create_buffer(const char *name, size_t size)
        {
            auto id = hasher(name);
            if (buffers.find(id) != buffers.end())
            {
                // resolve hash conflict
                id = hasher(name + std::to_string(size));
            }
            // check if the buffer already exists
            THROW_WHEN(buffers.find(id) != buffers.end(), "Buffer already exists: " + std::string(name));
            buffers[id] = stringBuf(name, size);
            return id;
        }
        stringBuf &get_buffer(size_t id)
        {
            THROW_WHEN(buffers.find(id) == buffers.end(), "Buffer not found: " + std::to_string(id));
            return buffers[id];
        }

    private:
        std::map<size_t, stringBuf> buffers;
        std::hash<std::string> hasher;
    };

    class fiostream
    {
    public:
        fiostream() = default;
        virtual ~fiostream() = default;
        void load_file_to_string_buffer(std::string filename, bufferManager &bm)
        {
            auto path = std::filesystem::absolute(filename);
            file.open(path);
            THROW_WHEN(!file.is_open(), "fiostream error: Can't open the file: " + path.string());

            // calculate file size
            file.seekg(0, std::ios::end);
            std::size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            auto id = bm.create_buffer(path.string().c_str(), file_size);
            auto &buf = bm.get_buffer(id);

            file.read(&buf[0], file_size);
        }

    private:
        std::ifstream file;
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