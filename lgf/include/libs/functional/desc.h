
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H

#include "lgf/value.h"

namespace lgf{

class real_t: public simpleValue {
    public:
    real_t() : simpleValue("real") {}
};

class set_desc : public simpleValue {
    public:
    set_desc() : simpleValue("set"){}
};

class empty_set_t: public simpleValue{
    public:
    empty_set_t() : simpleValue("empty-set") {};
};

class sigma_algebra_t : public simpleValue{ 
    public:
    sigma_algebra_t() : simpleValue("sigma-algebra") {}
};

} // namespace lgf

#endif