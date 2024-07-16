
#ifndef LGF_FUNCTIONAL_ANALYSIS_OPS_H
#define LGF_FUNCTIONAL_ANALYSIS_OPS_H

#include "lgf/node.h"
#include "libs/builtin/ops.h"
#include "desc.h"
#include "lgf/utils.h"

namespace lgf
{

    class mappingOp : public node
    {
    public:
        mappingOp() = default;
        mappingOp(std::string name) : node(name) { mark_status(eIdenticalRemovable); }
    };

    class elemFuncOp : public mappingOp
    {
    public:
        elemFuncOp(sid_t n) : mappingOp(n){};
    };

    class funcCosOp : public elemFuncOp
    {
    public:
        funcCosOp() : elemFuncOp("cos") {}
        static funcCosOp *build(LGFContext *ctx, node *x)
        {
            auto op = new funcCosOp();
            op->register_input(x);
            op->set_value_desc(x->get_value_desc());
            return op;
        }
        virtual sid_t represent() override final
        {
            return value_rep() + " = cos( " + input()->get_value_sid() + " )";
        }
    };

    class funcSineOp : public elemFuncOp
    {
    public:
        funcSineOp() : elemFuncOp("sine") {}
        static funcSineOp *build(LGFContext *ctx, node *x)
        {
            auto op = new funcSineOp();
            op->register_input(x);
            op->set_value_desc(x->get_value_desc());
            return op;
        }
        virtual sid_t represent() override
        {
            return value_rep() + " = sin( " + input()->get_value_sid() + " )";
        }
    };

    class funcPowerOp : public elemFuncOp
    {
    public:
        funcPowerOp() : elemFuncOp("power") {}
        static funcPowerOp *build(LGFContext *ctx, node *x, double n)
        {
            auto op = new funcPowerOp();
            op->register_input(x);
            op->set_power(n);
            op->set_value_desc(x->get_value_desc());
            return op;
        }
        void set_power(double n) { p = n; }
        double power() { return p; }
        double p = 1;
        virtual sid_t represent() override
        {
            return value_rep() + " = " + input()->get_value_sid() + " with power= " + utils::to_string(p);
        }
    };

    class funcExponentationOp : public elemFuncOp, public normalizer
    {
    public:
        funcExponentationOp() : elemFuncOp("functional::exp") {}
        static funcExponentationOp *build(LGFContext *ctx, node *base, node *power)
        {
            auto op = new funcExponentationOp();
            op->register_input(base, power);
            op->set_value_desc(power->get_value_desc());
            return op;
        }
        node *power()
        {
            return input(1);
        }
        node *base()
        {
            return input(0);
        }
        virtual sid_t represent() override
        {
            return value_rep() + " = exponent base: " + input()->get_value_sid() + " w. power: " + power()->get_value_sid();
        }
        virtual resultCode normalize(painter &p, node *op) override;
    };

    class funcLogarithmOp : public elemFuncOp, public normalizer
    {
    public:
        funcLogarithmOp() : elemFuncOp("functional::log") {}
        static funcLogarithmOp *build(LGFContext *ctx, node *base, node *arg)
        {
            auto op = new funcLogarithmOp();
            op->register_input(base, arg);
            op->set_value_desc(base->get_value_desc());
            return op;
        }
        node *base()
        {
            return input(0);
        }
        node *arg()
        {
            return input(1);
        }
        virtual sid_t represent() override
        {
            return value_rep() + " = log base: " + base()->get_value_sid() + " of " + arg()->get_value_sid();
        }
        virtual resultCode normalize(painter &p, node *op) override final;
    };

    class partialDifferentiateOp : public elemFuncOp
    {
    public:
        partialDifferentiateOp() : elemFuncOp("PartialDifferential") {}
        static partialDifferentiateOp *build(LGFContext *ctx, node *func, node *var, int order = 1)
        {
            auto op = new partialDifferentiateOp();
            op->register_input(func, var);
            op->set_order(order);
            op->set_value_desc(func->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *var() { return input(1); }
        void set_order(int r) { order = r; }
        int get_order() { return order; }
        virtual sid_t represent() override
        {
            auto res = value_rep() + " = Partial Differential";
            res = res + ": " + input()->get_value_sid() + " w.r.t. " + input(1)->get_value_sid() + ", order = " + std::to_string(order);
            return res;
        }

    private:
        int order = 1;
    };

    class differentiateOp : public mappingOp
    {
    public:
        differentiateOp() : mappingOp("Differential") {}
        static differentiateOp *build(LGFContext *ctx, node *input)
        {
            auto op = new differentiateOp();
            op->register_input(input);
            op->set_value_desc(input->get_value_desc());
            return op;
        }
        virtual sid_t represent() override
        {
            auto res = value_rep() + " = " + get_sid() + ": " + input()->get_value_sid();
            return res;
        }

    private:
        // int order = 1;
    };

    class RiemannIntegralOp : public node
    {
    public:
        RiemannIntegralOp() : node("Integral") {}
        static RiemannIntegralOp *build(LGFContext *ctx, node *integrand_, node *target_, node *low, node *high)
        {
            auto op = new RiemannIntegralOp();
            op->register_input(integrand_, target_, low, high);
            op->set_value_desc(integrand_->get_value_desc());
            return op;
        }
        node *get_integrand()
        {
            return input(0);
        }
        node *get_target()
        {
            return input(1);
        }
        node *get_lower_bound()
        {
            return input(2);
        }
        node *get_upper_bound()
        {
            return input(3);
        }
        virtual sid_t represent() override
        {
            auto res = get_value_sid() + " = Riemann Integral " + get_integrand()->get_value_sid() + " w.r.t. " + get_target()->get_value_sid() + " from " + get_lower_bound()->get_value_sid() + " to " + get_upper_bound()->get_value_sid();
            return res;
        }
    };

    class unionOp : public mappingOp
    {
    public:
        unionOp() : mappingOp("functional::union") {}
        template <typename... ARGS>
        static unionOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new unionOp();
            op->register_input(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
    };

    class intersectOp : public mappingOp
    {
    public:
        intersectOp() : mappingOp("functional::intersect") {}
        template <typename... ARGS>
        static intersectOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new intersectOp();
            op->register_input(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
    };

} // namespace lgf

#endif