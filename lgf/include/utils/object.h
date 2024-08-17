#ifndef UTILS_OBJECTS_H
#define UTILS_OBJECTS_H
#include <map>

namespace utils
{
    template <typename K, typename C>
    class cursive_map
    {
    public:
        cursive_map() = default;
        ~cursive_map() = default;
        void add(const K &key, const C &content)
        {
            if (_content_.find(key) != _content_.end())
            {
                return;
            }
            _content_[key] = content;
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
        std::map<K, cursive_map<K, C>> &get_map()
        {
            return _cmap_;
        }

    private:
        C _content_;
        std::map<K, cursive_map<K, C>> _cmap_;
    };
};

#endif