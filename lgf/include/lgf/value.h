#ifndef LGF_VALUE_H_
#define LGF_VALUE_H_
#include "object.h"
#include <string>
#include <vector>
#include "code.h"
namespace lgf
{

    class node;

    class descBase : public lgfObject
    {
    public:
        descBase() = default;
        descBase(sid_t id) : lgfObject(id) {}
        virtual ~descBase() = default;
        virtual sid_t represent() { return sid; }
    };

    class descriptor : public objectHandle<descBase>
    {
    public:
        descriptor() = default;
        descriptor(const descriptor &d) : objectHandle<descBase>(d) {}
        descriptor(std::shared_ptr<descBase> d) : objectHandle<descBase>(d) {}

        template <typename T>
        inline static descriptor get()
        {
            return descriptor(std::make_shared<T>());
        }
        template <typename T, typename... ARGS>
        inline static descriptor get(ARGS... args)
        {
            return descriptor(std::make_shared<T>(args...));
        }

        // std::unique_ptr<descBase> &get_desc_ptr() { return *(get_ptr()); }
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
        sid_t get_sid() { return value().get_sid(); }
    };

    class value : public lgfObject
    {
    public:
        value(std::string sid = "") : lgfObject(sid) {}

        value(descriptor &d, std::string sid = "") : lgfObject(sid), desc(d) {}

        virtual ~value() = default;
        virtual sid_t represent();
        sid_t desc_represent()
        {
            return desc.represent();
        }

        void print();

        descriptor get_desc()
        {
            return desc;
        }
        void set_desc(descriptor &d)
        {
            desc = d;
        }
        void set_nid(uint64_t id)
        {
            nid = id;
        }
        uint64_t get_nid() const { return nid; }

    private:
        uint64_t nid = 0;
        descriptor desc;
    };

} // namespace lgf

#endif