
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
            return value().represent();
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
}
#endif