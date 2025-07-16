#ifndef LGF_OBJECT_H_
#define LGF_OBJECT_H_
#include "base.h"
#include "attribute.h"
#include "graph.h"
#include <string>
#include <map>
#include <vector>
#include <iostream>

namespace lgf
{
    using namespace lgfc;
    // class object is an abstract concept representing any input/output of operations.
    // the details and properties of the object are defined by the description it holds.
    // the class description is a wrapper for the actual decription implementation inherited
    // from class descBase.
    class operation;
    class descBase
    {
    public:
        descBase() = default;
        descBase(std::string id) : name(id) {}
        virtual ~descBase() = default;
        virtual sid_t represent() { return name; }
        std::string get_name()
        {
            return name;
        }
        void set_attr(const std::string &key, const attribute &att)
        {
            attrs[key] = att;
        }
        attribute get_attr(const std::string &key)
        {
            auto it = attrs.find(key);
            if (it != attrs.end())
            {
                return it->second;
            }
            else
            {
                return attribute();
            }
        }
        bool has_attr(const std::string &key)
        {
            return attrs.find(key) != attrs.end();
        }

    private:
        std::string name;
        std::map<std::string, attribute> attrs; // attributes of the description
    };

    class description : public objectHandle<descBase>
    {
    public:
        description() = default;
        description(const description &d) : objectHandle<descBase>(d) {}
        description(std::shared_ptr<descBase> d) : objectHandle<descBase>(d) {}

        template <typename T>
        inline static description get()
        {
            return description(std::make_shared<T>());
        }
        template <typename T, typename... ARGS>
        inline static description get(ARGS... args)
        {
            return description(std::make_shared<T>(args...));
        }

        sid_t represent()
        {
            if (!is_null())
                return value().represent();
            else
                return "";
        }
        template <typename... ARGS>
        bool is()
        {
            return (... || dynamic_cast<ARGS *>(&value()));
        }
    };

    class object : public base, public node
    {
    public:
        object(std::string sid = "") : base(sid) {}

        object(const description &d, std::string sid = "") : base(sid), desc(d) {}

        virtual ~object() = default;
        virtual sid_t represent()
        {
            return get_sid() + " " + desc.represent();
        }

        description get_desc()
        {
            return desc;
        }
        void set_desc(description &d)
        {
            desc = d;
        }
        operation *get_defining_op();
        std::vector<operation *> get_users();

    private:
        description desc;
    };

} // namespace lgf

#endif