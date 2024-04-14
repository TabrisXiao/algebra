
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H

#include "lgf/value.h"

namespace lgf
{

    class real_t : public simpleValue
    {
    public:
        real_t() : simpleValue("real") {}
    };

    class set_desc : public simpleValue
    {
    public:
        set_desc() : simpleValue("set") {}
    };

    class realInterval : public set_desc
    {
    public:
        realInterval(double left, double right, bool o1, bool o2) : lb(left), rb(right), lop(o1), rop(o2)
        {
            set_sid("real-interval");
        }
        virtual sid_t represent() override
        {
            auto res = get_sid() + " ";
            std::string lbm = lop ? "(" : "[";
            std::string rbm = rop ? ")" : "]";
            res += lbm + lgf::utils::to_string(lb) + ", " + lgf::utils::to_string(rb) + rbm;
            return res;
        }
        bool is_belong(double x) const
        {
            if (x > rb)
                return false;
            if (x < lb)
                return false;
            if (x == rb && rop)
                return false;
            if (x == lb && lop)
                return false;
            return true;
        }

    private:
        double lb, rb;
        bool lop, rop;
    };

    class empty_set_t : public simpleValue
    {
    public:
        empty_set_t() : simpleValue("empty-set"){};
    };

    class sigma_algebra_t : public simpleValue
    {
    public:
        sigma_algebra_t() : simpleValue("sigma-algebra") {}
    };

} // namespace lgf

#endif