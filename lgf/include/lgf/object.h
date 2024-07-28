
#ifndef LGF_OBJECT_H_
#define LGF_OBJECT_H_
#include <memory>
#include "config.h"

namespace lgf
{

    class lgfObject
    {
    public:
        lgfObject() = default;
        lgfObject(sid_t id) : sid(id) {}
        std::string get_sid() const { return sid; }
        void set_sid(sid_t id) { sid = id; }
        bool set_sid_if_null(sid_t id)
        {
            if (sid.empty())
            {
                sid = id;
                return 1;
            }
            return 0;
        }
        template <typename... T>
        bool isa()
        {
            return (dynamic_cast<T *>(this) || ...);
        }
        template <typename T>
        T *dyn_cast()
        {
            return dynamic_cast<T *>(this);
        }

        virtual sid_t represent() = 0;

    protected:
        sid_t sid = "";
    };

    // bitCode is a template object to encode type tag into binary type:
    //
    template <typename digitType>
    class bitCode
    {
    public:
        using digit_t = digitType;
        bitCode() {}
        bitCode(bitCode &code) { value = code.value; }
        bitCode(const bitCode &code) { value = code.value; }
        bitCode(const digitType &val) { value = val; }

        bitCode shift(size_t val)
        {
            value |= 1 << val;
            return *this;
        }
        bool bit_check(size_t val)
        {
            return (value & (1 << val));
        }
        bitCode clear(size_t val)
        {
            value &= ~(1 << val);
            return *this;
        }
        bitCode add(const bitCode &val)
        {
            value |= val.value;
            return *this;
        }
        bool check(digitType val)
        {
            return (value & val) == val;
        }
        void reset() { value = 0; }
        digit_t value = 0;
    };

    // objectHandle is a template object to handle derived class of objectImp.
    // it resolve the class slicing problem by using shared_ptr to store the derived class.
    // The objectHandle object can be assigned to another objectHandle object.
    template <typename detial_t>
    class objectHandle
    {
    public:
        objectHandle() = default;
        objectHandle(std::shared_ptr<detial_t> &obj) : detail(obj) {}
        objectHandle(const objectHandle &obj) : detail(obj.detail) {}
        void assign(const std::shared_ptr<detial_t> &obj) { detail = obj; }
        bool is_null() const { return detail == nullptr; }
        objectHandle<detial_t> &operator=(const objectHandle &obj)
        {
            if (!obj.is_null())
                detail = obj.detail;
            return *this;
        }
        detial_t &value() { return *detail; }
        detial_t *ptr() { return detail.get(); }
        template <typename U>
        U *dyn_cast()
        {
            auto p = dynamic_cast<U *>(detail.get());
            return p;
        }

    private:
        std::shared_ptr<detial_t> detail;
    };
}

#endif