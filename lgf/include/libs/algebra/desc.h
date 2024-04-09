
#ifndef LGF_LIB_ALGEBRA_DESC_H
#define LGF_LIB_ALGEBRA_DESC_H

#include "lgf/value.h"
#include "lgf/attribute.h"

namespace lgf{

class realNumber : public simpleValue
{
    public:
    realNumber() : simpleValue("realNumber") {}
};

} // namespace lgf

#endif