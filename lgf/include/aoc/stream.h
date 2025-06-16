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
    // messager plays a role that writing messages to the buffer it handles.

    class stringBuf : public std::unique_ptr<std::string>
    {
    public:
        stringBuf() = default;
        ~stringBuf() = default;
        stringBuf(stringBuf &&) noexcept = default;
        stringBuf(std::string &&str) : std::unique_ptr<std::string>(std::make_unique<std::string>(std::move(str))) {}
        stringBuf(size_t size) : std::unique_ptr<std::string>(std::make_unique<std::string>(size, 0)) {}
        stringBuf &operator=(const stringBuf &) = delete;      // Disable copy assignment
        stringBuf &operator=(stringBuf &&) noexcept = default; // Enable move assignment
        size_t size() const
        {
            return this->get()->size();
        }
        const char *end() const
        {
            return &(*get())[size()];
        }
        char *begin() const
        {
            return get()->data();
        }
    };

    class stringBufferProducer
    {
    public:
        stringBufferProducer() = default;
        virtual ~stringBufferProducer() = default;
        stringBufferProducer &operator<<(const std::string &str)
        {
            ss << str;
            return *this;
        }
        void indent_level_up() { curIndentLevel++; }
        void indent_level_down()
        {
            if (curIndentLevel > 0)
                curIndentLevel--;
        }
        stringBufferProducer &indent()
        {
            for (int i = 0; i < curIndentLevel; i++)
                ss << "    ";
            return *this;
        }
        stringBufferProducer &incr_indent()
        {
            indent_level_up();
            return indent();
        }
        stringBufferProducer &decr_indent()
        {
            indent_level_down();
            return indent();
        }
        stringBuf write_to_buffer()
        {
            return stringBuf(ss.str());
        }
        void clear()
        {
            ss.clear();
        }

    private:
        std::ostringstream ss;
        size_t curIndentLevel = 0;
    };

    class fiostream
    {
    public:
        fiostream() = default;
        virtual ~fiostream() = default;
        stringBuf load_file_to_string_buffer(std::string filename)
        {
            auto path = std::filesystem::absolute(filename);
            ifs.open(path);
            THROW_WHEN(!ifs.is_open(), "fiostream error: Can't open the file: " + path.string());

            // calculate file size
            ifs.seekg(0, std::ios::end);
            std::size_t file_size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            auto buf = stringBuf(file_size);

            ifs.read(buf.begin(), file_size);
            ifs.close();
            return std::move(buf);
        }
        void write_string_buffer_to_file(const char *filename, const stringBuf &buf)
        {
            auto path = std::filesystem::absolute(filename);
            // create file if not exist
            if (!std::filesystem::exists(path.parent_path()))
            {
                std::filesystem::create_directories(path.parent_path());
            }
            ofs.open(path);
            THROW_WHEN(!ofs.is_open(), "fiostream error: Can't open the file: " + path.string());
            ofs.write(buf.begin(), buf.size());
        }

    private:
        std::ifstream ifs;
        std::ofstream ofs;
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