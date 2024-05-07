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
        virtual std::unique_ptr<descBase> copy() { return std::make_unique<descBase>(sid); };
        virtual sid_t represent() { return sid; }
    };

    class descriptor : public morphism_wrapper<descBase>
    {
    public:
        descriptor() = default;
        descriptor(const descriptor &d) : morphism_wrapper<descBase>(d) {}
        descriptor(std::unique_ptr<descBase> &&p)
        {
            ptr = std::move(p->copy());
        }
        template <typename T>
        inline static descriptor get()
        {
            return descriptor(std::make_unique<T>());
        }
        template <typename T, typename... ARGS>
        inline static descriptor get(ARGS... args)
        {
            return descriptor(std::make_unique<T>(args...));
        }

        std::unique_ptr<descBase> &get_desc_ptr() { return ptr; }
        sid_t represent()
        {
            if (ptr)
                return ptr->represent();
            else
                return "";
        }
        template <typename... ARGS>
        bool is()
        {
            return (... || dynamic_cast<ARGS *>(ptr.get()));
        }
        sid_t get_sid() { return ptr->get_sid(); }
    };

    class value : public lgfObject
    {
    public:
        value(std::string sid = "") : lgfObject(sid) {}

        value(descriptor &d, std::string sid = "") : lgfObject(sid), desc(std::move(d.get_desc_ptr())) {}

        virtual ~value() = default;
        virtual sid_t represent();
        sid_t desc_represent()
        {
            return desc.represent();
        }

        void print();

        descriptor get_desc() { return desc; }
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