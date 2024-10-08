
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/builtin/builtin.h"
#include "libs/math/functional/functional.h"
#include "variable.h"

namespace lgi::math
{
    using namespace lgi;
    variable cos(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::math::funcCosOp>(x.node());
        return variable(res);
    }

    variable sin(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::math::funcSineOp>(x.node());
        return variable(res);
    }

    template <typename T, typename U>
    variable exponent(const T &x, const U &y)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::math::funcExponentationOp>(x.node(), y.node());
        return variable(res);
    }

    variable exp(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto e = constant::e();
        auto res = canvas::get().get_painter().paint<lgf::math::funcExponentationOp>(e.node(), x.node());
        return variable(res);
    }

    variable log(const variable &base, const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::math::funcLogarithmOp>(base.node(), x.node());
        return variable(res);
    }

    variable ln(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto e = constant::e();
        auto res = canvas::get().get_painter().paint<lgf::math::funcLogarithmOp>(e.node(), x.node());
        return variable(res);
    }

    variable d(const variable &x)
    {
        auto res = canvas::get().get_painter().paint<lgf::math::differentiateOp>(x.node());
        return variable(res);
    }

    variable integral(const variable &f, const variable target, const variable &low, const variable &high)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::math::RiemannIntegralOp>(f.node(), target.node(), low.node(), high.node());
        return variable(res);
    }

    variable integral(const variable &f, const variable target, const double low, const double high)
    {
        variable l = low, h = high;
        return integral(f, target, l, h);
    };

    class set : public variableBase
    {
    public:
        set(bool init = 1)
        {
            if (!init)
                return;
            auto &ctx = canvas::get().get_context();
            v = canvas::get().get_painter().paint<lgf::declOp>(lgf::math::setDesc::get());
        }

        set(lgf::node *val) : variableBase(val) {}
    };

    // class interval : public set
    // {
    // public:
    //     interval(double lb, double rb, bool lop = 0, bool rop = 0) : set(false)
    //     {
    //         auto &ctx = canvas::get().get_context();
    //         auto &p = canvas::get().get_painter();
    //         v = p.paint<lgf::declOp>(lgf::realInterval::get(lb, rb, lop, rop));
    //     }
    // };

} // namespace  lgi::function

#endif