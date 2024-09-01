#ifndef AOC_OBJECTS_H
#define AOC_OBJECTS_H
#include <map>
#include <string>
#include <optional>
namespace aoc
{
    template <typename K, typename C>
    class cursive_map
    {
    public:
        cursive_map() = default;
        ~cursive_map() = default;
        cursive_map(const C &content)
        {
            _content_ = content;
        }
        void add(const K &key, const C &content)
        {
            if (!_cmap_.has_value())
            {
                _cmap_ = std::map<K, cursive_map<K, C>>();
            }
            _cmap_.value()[key] = cursive_map<K, C>(content);
        }
        bool has(const K &key)
        {
            if (!_cmap_.has_value())
            {
                return false;
            }
            return _cmap_.value().find(key) != _cmap_.value().end();
        }
        cursive_map<K, C> &get(const K &key)
        {
            if (has(key))
            {
                return _cmap_.value().at(key);
            }
            throw std::runtime_error("get: invalid key!");
        }
        C &get_value()
        {
            return _content_;
        }
        std::map<K, cursive_map<K, C>> &get_map()
        {
            if (_cmap_.has_value())
            {
                return _cmap_.value();
            }
            throw std::runtime_error("get_map: map doesn't exists!");
        }
        bool has_map()
        {
            return _cmap_.has_value();
        }

    private:
        C _content_;
        std::optional<std::map<K, cursive_map<K, C>>> _cmap_;
    };

    class stringRef
    {
    public:
        stringRef() = default;
        // stringRef(const char *p) : ptr(p), size(std::strlen(p)) {}
        stringRef(const char *p, size_t s) : ptr(p), size(s) {}
        stringRef(const std::string &s) : ptr(s.data()), size(s.size()) {}
        std::string data()
        {
            return std::string(ptr, size);
        }
        bool operator==(const stringRef &rhs) const
        {
            if (size != rhs.size)
                return false;
            return std::strncmp(ptr, rhs.ptr, size) == 0;
        }
        bool operator==(const std::string &rhs) const
        {
            if (size != rhs.size())
                return false;
            return std::strncmp(ptr, rhs.data(), size) == 0;
        }
        bool operator!=(const stringRef &rhs) const
        {
            return !(*this == rhs);
        }
        bool operator!=(const std::string &rhs) const
        {
            return !(*this == rhs);
        }
        const char *end() const { return ptr + size; }

    private:
        const char *ptr = nullptr;
        size_t size = 0;
    };

};

#endif