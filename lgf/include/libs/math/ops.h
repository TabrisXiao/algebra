

#ifndef LGF_LIB_MATH_OPS_H
#define LGF_LIB_MATH_OPS_H
#include "libs/builtin/ops.h"
#include "lgf/pass.h"

namespace lgf::math
{

    class mathOp : public node, public identiferInterface
    {
    public:
        mathOp(sid_t name) : node(name) { mark_status(eIdenticalRemovable); }
        ~mathOp() {}
    };
}

#endif