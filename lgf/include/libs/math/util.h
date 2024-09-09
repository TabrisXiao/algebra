
#ifndef LGF_LIB_MATH_UTIL_H
#define LGF_LIB_MATH_UTIL_H
#include "lgf/node.h"
#include "lgf/pass.h"
#include "libs/builtin/ops.h"
#include "desc.h"

namespace lgf::math
{
    inline node *create_real_constant(painter &p, double val)
    {
        auto attr = attribute::get<realNumberData>(realNumberData::real, val);
        auto desc = descriptor::get<realNumber>();
        return p.paint<cstDeclOp>(desc, attr);
    }

    inline node *create_real_pi(painter &p)
    {
        auto attr = attribute::get<realNumberData>(realNumberData::pi);
        auto desc = descriptor::get<realNumber>();
        return p.paint<cstDeclOp>(desc, attr);
    }

    inline node *create_real_e(painter &p)
    {
        auto attr = attribute::get<realNumberData>(realNumberData::e);
        auto desc = descriptor::get<realNumber>();
        return p.paint<cstDeclOp>(desc, attr);
    }

    inline bool is_cst_number(node *n, realNumberData val)
    {
        auto cst = n->dyn_cast<cstDeclOp>();
        if (!cst)
        {
            return false;
        }
        auto data = cst->value();
    }

    template <typename T>
    resultCode flatten_same_type_inputs(node *op)
    {
        resultCode result = resultCode::pass();
        size_t i = 0;
        while (i < op->get_input_size())
        {
            auto node = op->input(i);
            if (auto cast = node->dyn_cast<T>())
            {
                auto new_inputs = cast->get_input_nodes();
                op->drop_input(cast);
                op->register_inputs_at(new_inputs, i);
                if (cast->get_user_size() == 0)
                    cast->erase();
                result.add(resultCode::success());
            }
            i++;
        }
        return result;
    }

    template <typename callable, typename lhsTy, typename rhsTy>
    resultCode run_op_pair_base_on_op(node *op, callable &&fn)
    {
        resultCode result = resultCode::pass();
        size_t i = 1;
        while (i < op->get_input_size())
        {
            auto lhs = op->input(i - 1)->dyn_cast<lhsTy>();
            auto rhs = op->input(i)->dyn_cast<rhsTy>();
            if (lhs && rhs)
            {
                fn(lhs, rhs);
                result.add(resultCode::success());
            }
            else
            {
                lhs = op->input(i)->dyn_cast<lhsTy>();
                rhs = op->input(i - 1)->dyn_cast<rhsTy>();
                if (lhs && rhs)
                {
                    fn(lhs, rhs);
                    result.add(resultCode::success());
                }
            }
            i++;
        }
        return result;
    }

    template <typename lhsTy, typename rhsTy, typename callable>
    resultCode run_op_pair_base_on_desc(node *op, callable fn)
    {
        resultCode result = resultCode::pass();
        size_t i = 1;
        while (i < op->get_input_size())
        {
            auto lhs = op->input(i - 1);
            auto rhs = op->input(i);
            if (lhs->get_value_desc_as<lhsTy>() && rhs->get_value_desc_as<rhsTy>())
            {
                result.add(fn(lhs, rhs));
            }
            else if (lhs->get_value_desc_as<rhsTy>() && rhs->get_value_desc_as<lhsTy>())
            {
                result.add(fn(rhs, lhs));
            }
            i++;
        }
        return result;
    }

} // namespace lgf

#endif