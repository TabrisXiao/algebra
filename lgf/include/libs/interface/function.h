
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/functional/functional.h"
#include "variable.h"

namespace lgi::function
{
    using namespace lgi;
    variable cos(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::funcCosOp>(x.node());
        return variable(res);
    }

    variable sin(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::funcSineOp>(x.node());
        return variable(res);
    }

    variable power(const variable &x, double n)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::funcPowerOp>(x.node(), n);
        return variable(res);
    }

    variable exp(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::funcExpOp>(x.node());
        return variable(res);
    }

    variable d(const variable &x)
    {
        auto &ctx = canvas::get().get_context();
        auto res = canvas::get().get_painter().paint<lgf::differentiateOp>(x.node());
        return variable(res);
    }

    class set : public variableBase
    {
    public:
        set(bool init = 1)
        {
            if (!init)
                return;
            auto &ctx = canvas::get().get_context();
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<lgf::setDesc>());
        }

        set(lgf::node *val) : variableBase(val) {}
    };

    class interval : public set
    {
    public:
        interval(double lb, double rb, bool lop = 0, bool rop = 0) : set(false)
        {
            auto &ctx = canvas::get().get_context();
            auto &p = canvas::get().get_painter();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto lbv = p.paint<lgf::cstDeclOp>(real, ctx.get_data_attr<lgf::realNumberAttr>(lb));
            auto rbv = p.paint<lgf::cstDeclOp>(real, ctx.get_data_attr<lgf::realNumberAttr>(rb));
            v = p.paint<lgf::declOp>(ctx.get_desc<lgf::realInterval>(lbv, rbv, lop, rop), lbv, rbv);
        }
        interval(variable &x, variable &y)
        {
            auto &ctx = canvas::get().get_context();
            auto &p = canvas::get().get_painter();
            v = p.paint<lgf::declOp>(ctx.get_desc<lgf::realInterval>(x.node(), y.node(), 0, 0), x.node(), y.node());
        }
    };

} // namespace  lgi::function

#endif