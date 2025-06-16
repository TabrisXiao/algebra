#ifndef AOC_OBJECTS_H
#define AOC_OBJECTS_H
#include <map>
#include <string>

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
            if (_content_.find(key) != _content_.end())
            {
                return;
            }
            _cmap_[key] = cursive_map<K, C>(content);
        }
        bool has(const K &key)
        {
            return _cmap_.find(key) != _cmap_.end();
        }
        cursive_map<K, C> &get(const K &key)
        {
            if (has(key))
            {
                return _cmap_[key];
            }
            throw std::runtime_error("get: invalid key!");
        }
        C &get_value()
        {
            return _content_;
        }
        C &get_value(const K &key)
        {
            if (has(key))
            {
                return _cmap_[key].get_value();
            }
            throw std::runtime_error("get: invalid key!");
        }
        std::map<K, cursive_map<K, C>> &get_map()
        {
            return _cmap_;
        }

    private:
        C _content_;
        std::map<K, cursive_map<K, C>> _cmap_;
    };

    class stringRef
    {
    public:
        stringRef() = default;
        // stringRef(const char *p) : ptr(p), size(std::strlen(p)) {}
        stringRef(const char *p, size_t s) : ptr(p), size(s) {}
        stringRef(const std::string &s) : ptr(s.data()), size(s.size()) {}
        const char *data() const
        {
            return ptr;
        }
        std::string deepcopy()
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