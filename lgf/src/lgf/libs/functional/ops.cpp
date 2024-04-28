
#include "libs/functional/ops.h"
#include "libs/algebra/desc.h"

namespace lgf
{
    resultCode funcExponentationOp::rewrite(painter &p, node *op)
    {
        return resultCode::pass();
    }
    resultCode funcLogarithmOp::rewrite(painter &p, node *op)
    {
        auto ctx = p.get_context();
        auto log = op->dyn_cast<funcLogarithmOp>();
        auto base = log->base();
        auto arg = log->arg();
        if (base == arg)
        {
            auto unit = ctx->get_desc<unitDesc>(base->get_value_desc());
            p.replace_op<lgf::cstDeclOp>(op, unit);
            return resultCode::success();
        }
        return resultCode::pass();
    }
} // namespace lgf