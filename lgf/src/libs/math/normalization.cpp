
#include "libs/math/normalization.h"

using namespace lgf::math;
using namespace lgf;
bool is_unit(node *op)
{
    auto cst = op->dyn_cast<cstDeclOp>();
    if (!cst)
    {
        return false;
    }
    auto real = op->get_value_desc().dyn_cast<realNumber>();
    if (real)
    {
        if (auto data = cst->value().dyn_cast<realNumberData>())
        {
            return data->is_unit();
        }
        else if (auto fp32 = cst->value().dyn_cast<float32Data>())
        {
            return fp32->get_data() == 1.0;
        }
        else if (auto i32 = cst->value().dyn_cast<int32Data>())
        {
            return i32->get_data() == 1;
        }
        return false;
    }
    return false;
}
bool is_zero(node *op)
{
    auto cst = op->dyn_cast<cstDeclOp>();
    if (!cst)
    {
        return false;
    }
    auto real = op->get_value_desc().dyn_cast<realNumber>();
    if (real)
    {
        auto data = cst->value().dyn_cast<realNumberData>();
        if (!data)
        {
            return false;
        }
        return data->is_zero();
    }
    return false;
}

resultCode lgf::math::zeroRewriter::rewrite(painter &p, cstDeclOp *op)
{
    if (!is_zero(op))
    {
        return resultCode::pass();
    }
    resultCode res;
    for (auto &h : op->get_user_handles())
    {
        auto user = h.get_link_node();
        // if (user)
        //     std::cout << "user: " << user->represent() << std::endl;
        if (auto prod = dynamic_cast<productOp *>(user))
        {

            p.replace_op(prod, op);
            res.add(resultCode::success());
        }
        else if (auto sum = dynamic_cast<sumOp *>(user))
        {
            sum->drop_input(op);
            res.add(resultCode::success());
        }
    }
    return resultCode::pass();
}

resultCode lgf::math::unitRewriter::rewrite(painter &p, cstDeclOp *op)
{
    if (!is_unit(op))
    {
        return resultCode::pass();
    }
    resultCode res;
    for (auto &h : op->get_user_handles())
    {
        auto user = h.get_link_node();
        if (auto prod = dynamic_cast<productOp *>(user))
        {
            prod->drop_input(op);
            res.add(resultCode::success());
        }
    }
    return res;
}