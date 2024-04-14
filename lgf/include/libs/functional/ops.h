
#ifndef LGF_FUNCTIONAL_ANALYSIS_OPS_H
#define LGF_FUNCTIONAL_ANALYSIS_OPS_H

#include "lgf/node.h"
#include "libs/Builtin/ops.h"
#include "desc.h"

namespace lgf
{

    class mappingOp : public node
    {
    public:
        mappingOp() = default;
        mappingOp(std::string name) : node(name) {}

        template <typename... ARGS>
        void add_args(ARGS... args)
        {
            auto nds = std::initializer_list<node *>{args...};
            for (auto nd : nds)
            {
                register_input(nd);
                narg++;
            }
        }
        node *arg(int n = 0)
        {
            return input(n);
        }
        size_t get_arg_size()
        {
            return narg;
        }

    private:
        size_t narg = 0;
    };

    class funcSineOp : public mappingOp
    {
    public:
        funcSineOp() : mappingOp("sine") {}
        static funcSineOp *build(node *x)
        {
            auto op = new funcSineOp();
            op->add_args(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcCosOp : public mappingOp
    {
    public:
        funcCosOp() : mappingOp("cos") {}
        static funcCosOp *build(node *x)
        {
            auto op = new funcCosOp();
            op->add_args(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcPowerOp : public mappingOp
    {
    public:
        funcPowerOp() : mappingOp("power") {}
        static funcPowerOp *build(node *x, double n)
        {
            auto op = new funcPowerOp();
            op->add_args(x);
            op->set_power(n);
            op->infer_trivial_value_desc();
            return op;
        }
        void set_power(double n) { p = n; }
        double power() { return p; }
        double p = 1;
    };

    class funcExpOp : public mappingOp
    {
    public:
        funcExpOp() : mappingOp("functional::exp") {}
        static funcExpOp *build(node *power)
        {
            auto op = new funcExpOp();
            op->add_args(power);
            op->infer_trivial_value_desc();
            return op;
        }
        node *power()
        {
            return input(0);
        }
    };

    class partialDifferentiateOp : public mappingOp
    {
    public:
        partialDifferentiateOp() : mappingOp("PartialDifferentiate") {}
        static partialDifferentiateOp *build(node *func, node *var, int order = 1)
        {
            auto op = new partialDifferentiateOp();
            op->add_args(func, var);
            op->set_order(order);
            op->set_value_desc(func->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *var() { return input(1); }
        void set_order(int r) { order = r; }
        int get_order() { return order; }
        private:
        int order = 1;
    };

    class differentiateOp : public mappingOp
    {
    public:
        differentiateOp() : mappingOp("differentiate") {}
        static differentiateOp *build(node *input, node *target, int r = 1)
        {
            auto op = new differentiateOp();
            op->add_args(input, target);
            op->set_order(r);
            op->set_value_desc(input->get_value_desc());
            op->verify();
            return op;
        }
        void set_order(int r) { order = r; }
        int get_order() { return order; }
        node *func() { return input(0); }
        node *target() { return input(1); }
        virtual sid_t represent() override
        {
            auto res = get_value_sid() + " = " + get_sid();
            res = res + " : " + input()->get_value_sid() + " w.r.t. " + input(1)->get_value_sid() + ", order = " + std::to_string(order);
            return res;
        }
        void verify()
        {
            if (! dynamic_cast<lgf::declOp*>(arg(1)))
            {
                throw std::runtime_error("differentiateOp: the second argument can only be an independent variable");
            }
        }

    private:
        int order = 1;
    };

    class unionOp : public mappingOp
    {
    public:
        unionOp() : mappingOp("functional::union") {}
        template <typename... ARGS>
        static unionOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new unionOp();
            op->add_args(args...);
            op->createnode(op->inputnode()->getType());
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
            op->add_args(args...);
            op->createnode(op->inputnode()->getType());
            return op;
        }
    };

} // namespace lgf

#endif