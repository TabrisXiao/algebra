#include "libs/math/passes.h"
#include "libs/math/passes.h"
#include "libs/math/util.h"
namespace lgf::math
{

    bool is_1d_func_wrt_target(node *func, node *target)
    {
        // if (func->is_isomorphic(target))
        // {
        //     return true;
        // }
        if (func == target)
        {
            return true;
        }
        for (auto &input : func->get_input_nodes())
        {
            if (!is_1d_func_wrt_target(input, target))
            {
                return false;
            }
        }

        return true;
    }

    node *get_inverse_derivative(painter &p, node *func)
    {
        node *main = nullptr;
        if (func->isa<funcSineOp>())
        {
            auto cos = p.paint<funcCosOp>(func->input());
            main = p.paint<negativeOp>(cos);
        }
        else if (func->isa<funcCosOp>())
        {
            main = p.paint<funcSineOp>(func->input());
        }
        else if (func->isa<inverseOp>())
        {
            main = p.paint<funcLogarithmOp>(create_real_e(p), func->input());
        }
        else if (auto expo = func->dyn_cast<funcExponentationOp>())
        {
        }
        auto cst = p.paint<declOp>(func->input()->get_value_desc());
        node *res = nullptr;
        if (main)
            res = p.paint<sumOp>(main, cst);
        return res;
    }

    resultCode ReimannIntegralRewriter::rewrite(painter &p, RiemannIntegralOp *op)
    {
        auto result = resultCode::pass();
        node *func = op->get_integrand();
        node *target = op->get_target();
        if (func->isa<lgf::math::differentiateOp>())
        {
            if (is_1d_func_wrt_target(func->input(), target))
            {
                // integrate the derivative of a function
                auto candi = func->input();
                auto up = p.paint<lgf::math::approachOp>(candi, target, op->get_upper_bound());
                auto down = p.paint<lgf::math::approachOp>(candi, target, op->get_lower_bound());
                auto ng = p.paint<lgf::math::negativeOp>(down);
                p.replace_op<sumOp>(op, up, ng);
                return resultCode::success();
            }
        }
        return resultCode::pass();
    }
}
