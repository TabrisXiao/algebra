
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
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<lgf::set_desc>());
        }

        set(lgf::node *val) : variableBase(val) {}
    };

    class interval : public set
    {
    public:
        interval(double lb, double rb, bool lop, bool rop) : set(false)
        {
            auto &ctx = canvas::get().get_context();
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<lgf::realInterval>(lb, rb, lop, rop));
        }
        interval(lgf::node *val) : set(val) {}
    };

} // namespace  lgi::function

#endif