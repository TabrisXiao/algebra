
#ifndef LGF_OBJECT_H_
#define LGF_OBJECT_H_
#include "config.h"

namespace lgf
{

    class graphObject
    {
    public:
        graphObject() = default;
        graphObject(sid_t id) : sid(id) {}
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
        void set_nid(uint64_t id)
        {
            nid = id;
        }
        template <typename T>
        T *dyn_cast()
        {
            return dynamic_cast<T *>(this);
        }
        uint64_t get_nid() const { return nid; }
        virtual sid_t represent() = 0;

    protected:
        sid_t sid = "";
        uint64_t nid = 0;
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
}

#endif