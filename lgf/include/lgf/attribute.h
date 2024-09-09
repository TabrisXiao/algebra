
#ifndef LGF_ATTRIBUTE_H
#define LGF_ATTRIBUTE_H
#include "object.h"
#include <map>
namespace lgf
{
    class attrBase : public lgfObject
    {
    public:
        attrBase() = default;
        attrBase(sid_t id) : lgfObject(id) {}
        virtual ~attrBase() = default;
        virtual sid_t represent() = 0;
    };

    class attribute : public objectHandle<attrBase>
    {
    public:
        attribute() = default;
        attribute(const attribute &att) : objectHandle<attrBase>(att) {}
        attribute(std::shared_ptr<attrBase> imp) : objectHandle<attrBase>(imp)
        {
        }
        sid_t represent()
        {
            if (is_null())
                throw std::runtime_error("attribute: Invalid attribute!");
            return ptr()->represent();
        }
        template <typename T>
        inline static attribute get()
        {
            return attribute(std::make_shared<T>());
        }
        template <typename T, typename... ARGS>
        inline static attribute get(ARGS... args)
        {
            return attribute(std::make_shared<T>(args...));
        }
    };

    template <typename T>
    class singleData : public attrBase
    {
    public:
        singleData(sid_t id, T &t) : attrBase(id), data_(t) {}
        T get_data() { return data_; }

    private:
        T data_;
    };

    class tuple : public attrBase
    {
    public:
        std::vector<attribute> data;
    };

    class attrContainer
    {
    public:
        attrContainer() = default;
        virtual ~attrContainer() = default;
        attribute get_attr(sid_t id)
        {
            if (dict_attr.find(id) == dict_attr.end())
            {
                return attribute();
            }
            return dict_attr[id];
        }
        void add_attr(sid_t id, attribute att)
        {
            dict_attr[id] = att;
        }
        bool has(sid_t id)
        {
            return dict_attr.find(id) != dict_attr.end();
        }
        attribute attr(sid_t id)
        {
            return get_attr(id);
        }

        std::string default_attr_represent()
        {
            std::string res = "";
            if (dict_attr.empty())
                return res;
            for (auto &d : dict_attr)
            {
                res += d.first + ": ";
                res += d.second.represent();
                res += ", ";
            }
            res.substr(0, res.size() - 2);
            res = "{" + res + "}";
            return res;
        }

        std::map<sid_t, attribute> &get_dict()
        {
            return dict_attr;
        }
        std::map<sid_t, attribute> dict_attr;
    };

    class intAttr : public singleData<int>
    {
    public:
        intAttr(int t) : singleData<int>("integer", t) {}
        virtual sid_t represent()
        {
            return std::to_string(get_data());
        }
    };

    class F32Attr : public singleData<float>
    {
    public:
        F32Attr(float t) : singleData<float>("float32", t) {}
        virtual sid_t represent()
        {
            return std::to_string(get_data());
        }
    };
    class stringAttr : public singleData<std::string>
    {
    public:
        stringAttr(std::string t) : singleData<std::string>("string", t) {}
        virtual sid_t represent()
        {
            return get_data();
        }
    };

    class arraAttr : public attrBase
    {
    public:
        arraAttr(std::vector<attribute> t) : attrBase("array"), data(t) {}
        virtual sid_t represent()
        {
            sid_t res = "[";
            for (auto &d : data)
            {
                res += d.represent();
                res += ", ";
            }
            res.pop_back();
            res.pop_back();
            res += "]";
            return res;
        }
        std::vector<attribute> data;
    };

    template <typename T>
    class enumAttr : attrBase
    {
    public:
        enumAttr(sid_t id) : attrBase(id) {}
        enumAttr(sid_t id, T t) : attrBase(id), data(t) {}
        virtual ~enumAttr() = default;
        T value() { return data; }
        bool operator==(const enumAttr<T> &rhs) const
        {
            return data == rhs.value();
        }
        bool operator!=(const enumAttr<T> &rhs) const
        {
            return data != rhs.value();
        }
        static enumAttr<T> get(T t)
        {
            return enumAttr<T>(t);
        }

    private:
        T data;
    };

}
#endif