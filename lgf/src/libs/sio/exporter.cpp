
#include "libs/sio/exporter.h"
#include "libs/builtin/ops.h"
#include "libs/math/ops.h"
#include "libs/sio/ops.h"

namespace lgf::sio
{
    std::string export2latex::get_cst_represent(cstDeclOp *cst)
    {
        auto desc = cst->get_value_desc();
        auto data = cst->value();
        return data.represent();
    }

    std::string export2latex::process(node *n)
    {
        std::string res;
        if (auto op = dynamic_cast<math::productOp *>(n))
        {
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_link_node()) + "\\cdot ";
            }
            res = res.substr(0, res.size() - 6);
        }
        else if (auto op = dynamic_cast<math::sumOp *>(n))
        {
            res += "(";
            for (auto &input : op->get_input_handles())
            {
                auto temp = process(input.get_link_node());
                if (temp[0] == '-')
                {
                    res.pop_back();
                }
                res += temp + "+";
            }
            res.pop_back();
            res += ")";
        }
        else if (auto op = dynamic_cast<declOp *>(n))
        {
            auto sym = op->get_value_sid();
            if (sym[0] == '%')
            {
                sym[0] = '{';
                sym = "x_" + sym + "}";
            }
            res += sym;
        }
        else if (auto op = dynamic_cast<math::partialDifferentiateOp *>(n))
        {
            res = "\\frac{\\partial }{\\partial " + process(op->input(1)) + "} " + process(op->input(0));
        }
        else if (auto op = dynamic_cast<math::negativeOp *>(n))
        {
            res = "-" + process(op->input(0));
        }
        else if (auto op = dynamic_cast<math::differentiateOp *>(n))
        {
            res = "d" + process(op->input(0)) + " ";
        }
        else if (auto cst = n->dyn_cast<cstDeclOp>())
        {
            res = get_cst_represent(cst);
        }
        else if (auto inverse = n->dyn_cast<math::inverseOp>())
        {
            res = "1/" + process(inverse->input(0));
        }
        else if (auto cos = n->dyn_cast<math::funcCosOp>())
        {
            res = "\\cos(" + process(cos->input()) + ")";
        }
        else if (auto sine = n->dyn_cast<math::funcSineOp>())
        {
            res = "\\sin(" + process(sine->input()) + ")";
        }
        else if (auto power = n->dyn_cast<math::funcPowerOp>())
        {
            res = process(power->input()) + "^{" + process(power->input(1)) + "}";
        }
        else if (auto exp = n->dyn_cast<math::funcExponentationOp>())
        {
            res = "\\exp(" + process(exp->input()) + ")";
        }
        else if (auto log = n->dyn_cast<math::funcLogarithmOp>())
        {
            res = "\\log(" + process(log->input()) + ")";
        }
        else if (auto limit = n->dyn_cast<math::limitOp>())
        {
            res = "\\lim_{" + process(limit->var()) + "\\to " + process(limit->val()) + "} " + process(limit->func());
        }
        else
        {
            auto rep = "\n" + n->represent();
            THROW("The following Op is not supported in export2latex: " + rep + "\n");
        }
        return res;
    }

    // std::string export2latex::simp_func_expression(node *val)
    // {
    //     std::string result;
    //     auto funcName = val->dyn_cast<funcOp>()->getFuncName();
    //     if (funcName == "exp")
    //     {
    //         auto num = val->input()->dyn_cast<numberOp>();
    //         if (num && num->get_number_str() == "e")
    //         {
    //             return "\\exp\\left(" + process(val->input(1)) + "\\right)";
    //         }
    //         return process(val->input()) + "^{" + process(val->input(1)) + "}";
    //     }
    //     else if (funcName == "log")
    //     {
    //         if (is_special_number(val->input(), "e"))
    //         {
    //             return "\\ln\\left(" + process(val->input(1)) + "\\right)";
    //         }
    //         else
    //         {
    //             return "\\log_{" + process(val->input()) + "}\\left(" + process(val->input(1)) + "\\right)";
    //         }
    //     }

    //     result += funcName + "(";
    //     for (auto &input : val->get_input_handles())
    //     {
    //         result += process(input.get_link_node()) + ",";
    //     }
    //     result.pop_back();
    //     return result + ")";
    // }
} // namespace lgf::sio