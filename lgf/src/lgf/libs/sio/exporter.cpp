
#include "libs/sio/exporter.h"
#include "libs/builtin/ops.h"
#include "libs/sio/ops.h"

namespace lgf::sio
{
    std::string export2latex::process(node *n)
    {
        std::string res;
        if (auto op = dynamic_cast<sio::scalarProductOp *>(n))
        {
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_dual_node()) + "\\cdot ";
            }
            res = res.substr(0, res.size() - 6);
        }
        else if (auto op = dynamic_cast<sumOp *>(n))
        {
            res += "(";
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_dual_node()) + "+";
            }
            res.pop_back();
            res += ")";
        }
        else if (auto op = dynamic_cast<funcOp *>(n))
        {
            res += simp_func_expression(op);
        }
        else if (auto op = dynamic_cast<symbolOp *>(n))
        {
            auto sym = op->get_symbol();
            if (sym[0] == '%')
            {
                sym[0] = '{';
                sym = "x_" + sym + "}";
            }
            res += sym;
        }
        else if (auto op = dynamic_cast<partialD *>(n))
        {
            res = "\\frac{\\partial }{\\partial " + process(op->input(1)) + "} " + process(op->input(0));
        }
        else if (auto op = dynamic_cast<negativeOp *>(n))
        {
            res = "-" + process(op->input(0));
        }
        else if (auto op = dynamic_cast<differentialOp *>(n))
        {
            res = "d" + process(op->input(0)) + " ";
        }
        else if (auto cst = n->dyn_cast<numberOp>())
        {
            res = cst->get_number_str();
        }
        else if (auto inverse = n->dyn_cast<inverseOp>())
        {
            res = "1/" + process(inverse->input(0));
        }
        else
        {
            auto rep = "\n" + n->represent();
            THROW("The following Op is not supported in export2latex: " + rep + "\n");
        }
        return res;
    }

    bool is_special_number(node *op, std::string str)
    {
        auto num = op->dyn_cast<numberOp>();
        if (!num)
        {
            return false;
        }
        return num->get_number_str() == str;
    }

    std::string export2latex::simp_func_expression(node *val)
    {
        std::string result;
        auto funcName = val->dyn_cast<funcOp>()->getFuncName();
        if (funcName == "exp")
        {
            auto num = val->input()->dyn_cast<numberOp>();
            if (num && num->get_number_str() == "e")
            {
                return "\\exp\\left(" + process(val->input(1)) + "\\right)";
            }
            return process(val->input()) + "^{" + process(val->input(1)) + "}";
        }
        else if (funcName == "log")
        {
            if (is_special_number(val->input(), "e"))
            {
                return "\\ln\\left(" + process(val->input(1)) + "\\right)";
            }
            else
            {
                return "\\log_{" + process(val->input()) + "}\\left(" + process(val->input(1)) + "\\right)";
            }
        }

        result += funcName + "(";
        for (auto &input : val->get_input_handles())
        {
            result += process(input.get_dual_node()) + ",";
        }
        result.pop_back();
        return result + ")";
    }
} // namespace lgf::sio