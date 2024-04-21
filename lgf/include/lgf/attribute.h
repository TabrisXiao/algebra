
#ifndef LGF_ATTRIBUTE_H
#define LGF_ATTRIBUTE_H
#include "object.h"
namespace lgf
{
    class dataAttr
    {
    public:
        dataAttr() = default;
        dataAttr(sid_t id) : sid(id){};
        virtual sid_t represent() = 0;
        sid_t get_sid()
        {
            return sid;
        }

    private:
        sid_t sid;
    };

    // preserved data attribute contains contains the data itself.
    // this is useful when cost of data copy is low and make the
    // attribute creataion more convenient.
    template <typename T>
    class preservedDataAttr : public dataAttr
    {
    public:
        preservedDataAttr(sid_t id, T d) : data(d), dataAttr(id){};
        virtual sid_t represent()
        {
            return get_sid() + ", " + represent_data();
        }
        virtual sid_t represent_data() = 0;
        void set_data(T d)
        {
            data = d;
        }
        T get_data()
        {
            return data;
        }

    private:
        T data;
    };
}
#endif