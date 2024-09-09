

#ifndef LGF_LIB_MATH_OPS_H
#define LGF_LIB_MATH_OPS_H
#include "libs/builtin/ops.h"
#include "lgf/pass.h"
#include "desc.h"

namespace lgf::math
{

    class mathOp : public node, public identifier
    {
    public:
        mathOp(sid_t name) : node(name) { mark_status(eIdenticalRemovable); }
        ~mathOp() {}
    };

    class sumOp : public node, public normalizer
    {
    public:
        sumOp() : node("sum")
        {
            mark_status(eIdenticalRemovable);
            set_commutable(1);
        }
        static sumOp *build(LGFContext *ctx, std::vector<node *> &vec)
        {
            auto op = new sumOp();
            op->register_inputs(vec);
            auto desc = vec[0]->get_value_desc();
            op->set_value_desc(desc);
            return op;
        }
        template <typename... Args>
        static sumOp *build(LGFContext *ctx, Args... args)
        {
            auto op = new sumOp();
            op->register_input(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
        virtual resultCode normalize(painter &p, node *op);
    };

    class productOp : public node, public normalizer
    {
    public:
        productOp() : node("product")
        {
            mark_status(eIdenticalRemovable);
            set_commutable(1);
        }
        void inferOutput()
        {
            auto input_desc = input(0)->get_value_desc();
            if (input_desc.is<matrix>())
            {
                auto a_desc = input_desc.dyn_cast<matrix>();
                auto b_desc = input(1)->get_value_desc().dyn_cast<matrix>();
                auto output_desc = descriptor::get<matrix>(
                    a_desc->get_dim(0),
                    b_desc->get_dim(1),
                    a_desc->get_elem_desc());
            }
            else
                set_value_desc(input_desc);
        }
        static productOp *build(LGFContext *ctx, std::vector<node *> &vec)
        {
            auto op = new productOp();
            op->register_inputs(vec);
            op->inferOutput();
            return op;
        }
        static productOp *build(LGFContext *ctx)
        {
            auto op = new productOp();
            return op;
        }
        template <typename... Args>
        static productOp *build(LGFContext *ctx, Args... args)
        {
            auto op = new productOp();
            op->register_input(args...);
            op->inferOutput();
            return op;
        }
        virtual resultCode normalize(painter &p, node *op);
    };

    class negativeOp : public node
    {
    public:
        negativeOp() : node("negative") { mark_status(eIdenticalRemovable); }
        static negativeOp *build(LGFContext *ctx, node *n)
        {
            auto op = new negativeOp();
            op->register_input(n);
            op->set_value_desc(n->get_value_desc());
            return op;
        }
    };

    class inverseOp : public node
    {
    public:
        inverseOp() : node("inverse") { mark_status(eIdenticalRemovable); }
        static inverseOp *build(LGFContext *ctx, node *n)
        {
            auto op = new inverseOp();
            op->register_input(n);
            op->set_value_desc(n->get_value_desc());
            return op;
        }
    };

    class minusOp : public node
    {
    public:
        minusOp() : node("minus") { mark_status(eIdenticalRemovable); }
        static minusOp *build(LGFContext *ctx, node *lhs, node *rhs)
        {
            auto op = new minusOp();
            op->register_input(lhs, rhs);
            op->set_value_desc(lhs->get_value_desc());
            return op;
        }
        node *lhs() { return input(0); }
        node *rhs() { return input(1); }
        virtual sid_t represent() override
        {
            printer p;
            p << value_rep();
            p << " = " << get_sid() << ": " << lhs()->get_value_sid() << " - " << rhs()->get_value_sid();
            return p.dump();
        }
    };

    class elemFuncOp : public mathOp
    {
    public:
        elemFuncOp(sid_t n) : mathOp(n) {};
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
        static funcPowerOp *build(LGFContext *ctx, node *x, attribute power)
        {
            auto op = new funcPowerOp();
            op->register_input(x);
            op->add_attr("power", power);
            op->set_value_desc(x->get_value_desc());
            return op;
        }
        void set_power(attribute p) { add_attr("power", p); }
        float power() { return attr("power").dyn_cast<F32Attr>()->get_data(); }
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

    class approachOp : public mathOp
    {
    private:
        approachOp() : mathOp("approach") {}

    public:
        static approachOp *build(LGFContext *ctx, node *func, node *var, node *val)
        {
            auto op = new approachOp();
            op->register_input(func, var, val);
            op->set_value_desc(func->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *var() { return input(1); }
        node *val() { return input(2); }
        virtual sid_t represent() override
        {
            auto res = value_rep() + " = " + get_sid() + ": " + func()->get_value_sid() + " w.r.t. " + var()->get_value_sid() + " -> " + val()->get_value_sid();
            return res;
        }
    };

    class differentiateOp : public mathOp
    {
    public:
        differentiateOp() : mathOp("Differential") {}
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

    class unionOp : public mathOp
    {
    public:
        unionOp() : mathOp("functional::union") {}
        template <typename... ARGS>
        static unionOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new unionOp();
            op->register_input(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
    };

    class intersectOp : public mathOp
    {
    public:
        intersectOp() : mathOp("functional::intersect") {}
        template <typename... ARGS>
        static intersectOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new intersectOp();
            op->register_input(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
    };
}

#endif