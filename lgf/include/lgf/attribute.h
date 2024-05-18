
#ifndef LGF_ATTRIBUTE_H
#define LGF_ATTRIBUTE_H
#include "object.h"
namespace lgf
{
    class attrBase : public lgfObject
    {
    public:
        attrBase() = default;
        attrBase(sid_t id) : lgfObject(id) {}
        virtual ~attrBase() = default;
        virtual std::unique_ptr<attrBase> copy() = 0;
        virtual sid_t represent() = 0;
    };

    class attribute : public morphism_wrapper<attrBase>
    {
    public:
        attribute() = default;
        attribute(const attribute &att) : morphism_wrapper<attrBase>(att) {}
        attribute(std::unique_ptr<attrBase> &&imp)
        {
            set_ptr(std::move(imp));
        }
        sid_t represent()
        {
            if (!get_ptr())
                throw std::runtime_error("attribute: Invalid attribute!");
            return get_ptr()->represent();
        }
        template <typename T>
        inline static attribute get()
        {
            return attribute(std::make_unique<T>());
        }
        template <typename T, typename... ARGS>
        inline static attribute get(ARGS... args)
        {
            return attribute(std::make_unique<T>(args...));
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
}
#endif