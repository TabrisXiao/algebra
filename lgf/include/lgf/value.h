#ifndef LGF_VALUE_H_
#define LGF_VALUE_H_
#include "object.h"
#include <string>
#include <vector>

namespace lgf
{

    class node;
    class valueDesc : public graphObject
    {
    public:
        valueDesc() = default;
        valueDesc(sid_t id) : graphObject(id) {}
        virtual sid_t represent() { return ""; }
    };

    class value : public graphObject
    {
    public:
        value(std::string sid = "") : graphObject(sid) {}

        value(valueDesc *d, std::string sid = "") : graphObject(sid), desc(d) {}
        value(const value &v) : graphObject(v.get_sid()), desc(v.get_desc()) {}

        virtual ~value() = default;
        virtual sid_t represent();
        sid_t desc_represent()
        {
            if (desc)
                return desc->represent();
            return "";
        }

        void print();

        valueDesc *get_desc() const { return desc; }
        void set_desc(valueDesc *i)
        {
            desc = i;
        }

    private:
        valueDesc *desc = nullptr;
    };

    // simple value is the value that can be identified by a single sid.
    class simpleValue : public valueDesc
    {
    public:
        simpleValue(sid_t id) : valueDesc(id) {}
        virtual sid_t represent() override
        {
            return get_sid();
        }
    };
} // namespace lgf

#endif