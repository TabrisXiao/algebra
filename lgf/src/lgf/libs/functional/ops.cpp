#include "libs/algebra/ops.h"
#include "libs/functional/ops.h"
#include "libs/algebra/desc.h"

namespace lgf
{
    resultCode funcExponentationOp::rewrite(painter &p, node *op)
    {

        auto base = op->input(0);
        auto exp = base->dyn_cast<funcExponentationOp>();
        if (exp)
        {
            // merge (a^x)^y to a^(x*y)
            auto power = exp->input(1);
            auto new_exp = p.replace_op<funcExponentationOp>(op, exp->input(0), p.paint<productOp>(exp->input(1), op->input(1)));
            return resultCode::success();
        }

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
            auto unit = realNumber::get();
            auto unitAttr = ctx->get_data_attr<realNumberAttr>(realNumberAttr::one);
            p.replace_op<lgf::cstDeclOp>(op, unit, unitAttr);
            return resultCode::success();
        }
        return resultCode::pass();
    }
} // namespace lgf