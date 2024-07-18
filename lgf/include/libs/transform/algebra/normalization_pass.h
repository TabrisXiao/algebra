
#ifndef TRANSFORM_ALGEBRA_SIMPLIFICATION_PASS
#define TRANSFORM_ALGEBRA_SIMPLIFICATION_PASS
#include "libs/builtin/passes.h"
#include "libs/algebra/algebra.h"
namespace lgf
{
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
            if (auto data = cst->get_data_attr().dyn_cast<realNumberData>())
            {
                return data->is_unit();
            }
            else if (auto fp32 = cst->get_data_attr().dyn_cast<float32Data>())
            {
                return fp32->get_data() == 1.0;
            }
            else if (auto i32 = cst->get_data_attr().dyn_cast<int32Data>())
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
            auto data = cst->get_data_attr().dyn_cast<realNumberData>();
            if (!data)
            {
                return false;
            }
            return data->is_zero();
        }
        return false;
    }

    class zeroRewriter : public rewriter<cstDeclOp>
    {
    public:
        zeroRewriter() = default;
        virtual resultCode rewrite(painter &p, cstDeclOp *op);
    };

    resultCode zeroRewriter::rewrite(painter &p, cstDeclOp *op)
    {
        if (!is_zero(op))
        {
            return resultCode::pass();
        }
        resultCode res;
        for (auto &h : op->get_user_handles())
        {
            auto user = h.get_dual_node();
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

    class unitRewriter : public rewriter<cstDeclOp>
    {
    public:
        unitRewriter() = default;
        virtual resultCode rewrite(painter &p, cstDeclOp *op);
    };

    resultCode unitRewriter::rewrite(painter &p, cstDeclOp *op)
    {
        if (!is_unit(op))
        {
            return resultCode::pass();
        }
        resultCode res;
        for (auto &h : op->get_user_handles())
        {
            auto user = h.get_dual_node();
            if (auto prod = dynamic_cast<productOp *>(user))
            {
                prod->drop_input(op);
                res.add(resultCode::success());
            }
        }
        return res;
    }

    class algebraNormalizationPass : public normalizationPass
    {
    public:
        algebraNormalizationPass() : normalizationPass("algebra-normalization") {}
        virtual resultCode run() final
        {
            painter p(get_graph());
            add_rewriter<normalizeRewriter>();
            add_rewriter<zeroRewriter>();
            add_rewriter<unitRewriter>();
            remove_identical_ops(p, get_graph());
            auto result = apply_rewriter_greedy(p, get_graph());
            remove_identical_ops(p, get_graph());
            remove_unused_ops(get_graph());
            return result;
        }
    };

    std::unique_ptr<passBase> createAlgebraNormalizationPass()
    {
        return std::make_unique<algebraNormalizationPass>();
    }
} // namespace lgf
#endif
