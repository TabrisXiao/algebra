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
};

#endif