
#include "libs/functional/passes.h"
#include "libs/algebra/desc.h"

lgf::resultCode lgf::ChainRuleRewriter::rewrite(painter &p, differentiateOp *op)
{
    // this apply the differentiation chain rule to all
    // differentiateOp in the graph
    auto result = resultCode::pass();

    node *func = op->input();
    auto ctx = p.get_context();

    if (auto f = dynamic_cast<declOp *>(func))
    {
        return result;
    }

    if (dynamic_cast<differentiateOp *>(func) ||
        dynamic_cast<partialDifferentiateOp *>(func))
    {
        return resultCode::pass();
    }

    if (auto cst = func->dyn_cast<cstDeclOp>())
    {
        p.replace_op<cstDeclOp>(op, ctx->get_desc<zeroDesc>(cst->get_value_desc()));
        return resultCode::success();
    }

    if (auto sum = dynamic_cast<sumOp *>(func))
    {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
            if (!h.is_coupled())
                continue;
            auto arg = h.get_dual_node();
            auto df = p.paint<differentiateOp>(arg);
            sum_args.push_back(df);
        }
        p.replace_op<sumOp>(op, sum_args);
        return resultCode::success();
    }

    if (auto product = dynamic_cast<productOp *>(func))
    {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
            auto arg = h.get_dual_node();
            auto df = p.paint<differentiateOp>(arg);
            auto prod = p.paint<productOp>();
            for (auto &h1 : func->get_input_handles())
            {
                if (&h == &h1)
                {
                    prod->register_input(df);
                }
                else
                {
                    prod->register_input(h1.get_dual_node());
                }
            }
            prod->infer_trivial_value_desc();
            sum_args.push_back(prod);
        }
        p.replace_op<sumOp>(op, sum_args);
        return resultCode::success();
    }

    if (auto mapping = dynamic_cast<elemFuncOp *>(func))
    {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
            auto arg = h.get_dual_node();
            auto Df = p.paint<partialDifferentiateOp>(func, arg);
            auto dx = p.paint<differentiateOp>(arg);
            auto product = p.paint<productOp>(Df, dx);
            sum_args.push_back(product);
        }
        if (sum_args.size() == 1)
        {
            p.replace_op(op, sum_args[0]);
        }
        else
            p.replace_op<sumOp>(op, sum_args);
        result.add(resultCode::success());
    }

    return result;
}

lgf::resultCode lgf::analyticFuncDerivativeRewriter::rewrite(painter &p, partialDifferentiateOp *op)
{
    auto result = resultCode::pass();
    ctx = p.get_context();
    auto func = op->func();
    auto target = op->var();
    if (dynamic_cast<cstDeclOp *>(func) || dynamic_cast<cstDeclOp *>(target))
    {
        p.replace_op<cstDeclOp>(
            op, p.get_context()->get_desc<zeroDesc>(op->get_value_desc()));
        return resultCode::success();
    }

    if (func == target)
    {
        p.replace_op<cstDeclOp>(
            op, p.get_context()->get_desc<unitDesc>(target->get_value_desc()));
        return resultCode::success();
    }

    // case of dy/dx
    auto declop = func->dyn_cast<declOp>();
    if (declop)
    {
        if (declop == target)
        {
            p.replace_op<cstDeclOp>(
                op, p.get_context()->get_desc<unitDesc>(target->get_value_desc()));
            return resultCode::success();
        }
        else
        {
            p.replace_op<cstDeclOp>(
                op, p.get_context()->get_desc<zeroDesc>(func->get_value_desc()));
            return resultCode::success();
        }
    }

    if (auto sum = dynamic_cast<sumOp *>(func))
    {
        for (auto &h : func->get_input_handles())
        {
            if (!h.is_coupled())
                continue;
            auto arg = h.get_dual_node();
            auto df = p.paint<partialDifferentiateOp>(arg, target);
            sum->replace_input_by(h, df);
        }
        return resultCode::success();
    }

    if (auto product = dynamic_cast<productOp *>(func))
    {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
            auto arg = h.get_dual_node();
            auto df = p.paint<partialDifferentiateOp>(arg, target);
            auto prod = p.paint<productOp>();
            for (auto &h1 : func->get_input_handles())
            {
                if (&h == &h1)
                {
                    prod->register_input(df);
                }
                else
                {
                    prod->register_input(arg);
                }
            }
            prod->infer_trivial_value_desc();
            sum_args.push_back(prod);
        }
        p.replace_op<sumOp>(op, sum_args);

        return resultCode::success();
    }
    if (auto neg = func->dyn_cast<negativeOp>())
    {
        auto x = neg->input();
        auto dx = p.paint<partialDifferentiateOp>(x, target);
        auto neg_dx = p.paint<negativeOp>(dx);
        p.replace_op(op, neg_dx);
        return resultCode::success();
    }

    if (auto cos = func->dyn_cast<funcCosOp>())
    {
        auto x = cos->input();
        auto dx = p.paint<partialDifferentiateOp>(x, target);
        auto sin = p.paint<funcSineOp>(x);
        auto neg = p.paint<negativeOp>(sin);
        auto product = p.paint<productOp>(neg, dx);
        p.replace_op(op, product);
        return resultCode::success();
    }
    if (auto sin = func->dyn_cast<funcSineOp>())
    {
        auto x = sin->input();
        auto dx = p.paint<partialDifferentiateOp>(x, target);
        auto cos = p.paint<funcCosOp>(x);
        auto product = p.paint<productOp>(cos, dx);
        p.replace_op(op, product);
        return resultCode::success();
    }
    if (auto exp = func->dyn_cast<funcExponentationOp>())
    {
        // d x^y = y x^(y-1) dx + ln(x) x^y dy
        auto base = exp->base();
        auto power = exp->power();
        auto e_data = ctx->get_data_attr<realNumberAttr>(realNumberAttr::e);
        auto e = p.paint<cstDeclOp>(ctx->get_desc<realNumber>(), e_data);
        auto unit = p.paint<cstDeclOp>(ctx->get_desc<unitDesc>(target->get_value_desc()));
        auto nunit = p.paint<negativeOp>(unit);
        auto ym1 = p.paint<sumOp>(power, nunit);
        // y x^(y-1) dx
        auto dx = p.paint<partialDifferentiateOp>(base, target);
        auto xym1 = p.paint<funcExponentationOp>(base, ym1);
        auto product1 = p.paint<productOp>(power, xym1, dx);

        // ln(x) x^y dy
        auto ln = p.paint<funcLogarithmOp>(e, base);
        auto dy = p.paint<partialDifferentiateOp>(power, target);
        auto product2 = p.paint<productOp>(ln, exp, dy);
        // summation
        auto sum = p.paint<sumOp>(product1, product2);
        p.replace_op(op, sum);
        return resultCode::success();
    }
    return result;
}