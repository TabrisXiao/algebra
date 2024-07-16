
#ifndef LGF_OBJECT_H_
#define LGF_OBJECT_H_
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

    template <typename T>
    class morphism_wrapper
    {
        // this object is a container to wrapper all classes derived from base type T.
        // This object allow the copy and the assignment for the derived class provided the
        // wrapped class having a virtual function defined in the signature:
        // virtual std::unique_ptr<T> copy();
    public:
        morphism_wrapper() = default;

        morphism_wrapper(const morphism_wrapper &src)
        {
            ptr = std::move(src.get_ptr()->copy());
        }
        morphism_wrapper(morphism_wrapper &src)
        {
            ptr = std::move(src.get_ptr()->copy());
        }
        morphism_wrapper &operator=(const morphism_wrapper &src)
        {
            ptr = std::move(src.get_ptr()->copy());
            return *this;
        }
        T *get_ptr() const { return ptr.get(); }
        void set_ptr(std::unique_ptr<T> &&p)
        {
            ptr = std::move(p);
        }
        template <typename U>
        U *dyn_cast()
        {
            auto p = dynamic_cast<U *>(ptr.get());
            return p;
        }
        bool is_null() { return ptr == nullptr; }

    private:
        std::unique_ptr<T> ptr = nullptr;
    };
}

#endif