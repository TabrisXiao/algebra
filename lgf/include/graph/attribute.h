
#ifndef LGF_ATTRIBUTE_H
#define LGF_ATTRIBUTE_H
#include "base.h"
#include <vector>
namespace lgf
{
    // attributes are used to store the information of object descriptions or operations.
    // class attribute is a wrapper for the attrBase class.
    // All the actual implementations of attributes should inherited from attrBase.
    using namespace lgfc;
    class attrBase
    {
    public:
        attrBase() = default;
        attrBase(std::string id) : name(id) {}
        virtual ~attrBase() = default;
        virtual sid_t represent() = 0;
        std::string get_name()
        {
            return name;
        }

    private:
        const std::string name;
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
        singleData(std::string id, T &t) : attrBase(id), data_(t) {}
        T get_data() { return data_; }
        virtual sid_t to_sid() = 0;
        sid_t represent() override
        {
            return attrBase::represent() + ":" + to_sid();
        }

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