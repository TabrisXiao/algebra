
#ifndef LGF_FUNCTIONAL_ANALYSIS_OPS_H
#define LGF_FUNCTIONAL_ANALYSIS_OPS_H

#include "lgf/node.h"
#include "desc.h"

namespace lgf
{

    class funcSineOp : public node
    {
    public:
        funcSineOp() : node("sine") {}
        static funcSineOp *build(node *x)
        {
            auto op = new funcSineOp();
            op->register_input(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcCosOp : public node
    {
    public:
        funcCosOp() : node("cos") {}
        static funcCosOp *build(node *x)
        {
            auto op = new funcCosOp();
            op->register_input(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcPowerOp : public node
    {
    public:
        funcPowerOp() : node("power") {}
        static funcPowerOp *build(node *x, double n)
        {
            auto op = new funcPowerOp();
            op->register_input(x);
            op->setPower(n);
            op->infer_trivial_value_desc();
            return op;
        }
        void setPower(double n) { p = n; }
        double power() { return p; }
        double p = 1;
    };

    class funcExpOp : public node
    {
    public:
        funcExpOp() : node("functional::exp") {}
        static funcExpOp *build(node *power)
        {
            auto op = new funcExpOp();
            op->register_input(power);
            op->infer_trivial_value_desc();
            return op;
        }
        node *power()
        {
            return input(0);
        }
    };

    class partialDifferentiateOp : public node
    {
    public:
        partialDifferentiateOp() : node("PartialDifferentiate") {}
        static partialDifferentiateOp *build(node *func, node *var)
        {
            auto op = new partialDifferentiateOp();
            op->register_input(func, var);
            op->set_value_desc(func->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *var() { return input(1); }
    };

    class differentiateOp : public node
    {
    public:
        differentiateOp() : node("differentiate") {}
        static differentiateOp *build(LGFContext *ctx, node *input, node *target)
        {
            auto op = new differentiateOp();
            op->register_input(input, target);
            op->set_value_desc(input->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *target() { return input(1); }
    };

    class unionOp : public node
    {
    public:
        unionOp() : node("functional::union") {}
        template <typename... ARGS>
        static unionOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new unionOp();
            op->registerInput(args...);
            op->createnode(op->inputnode()->getType());
            return op;
        }
    };

    class intersectOp : public node
    {
    public:
        intersectOp() : node("functional::intersect") {}
        template <typename... ARGS>
        static intersectOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new intersectOp();
            op->registerInput(args...);
            op->createnode(op->inputnode()->getType());
            return op;
        }
    };

} // namespace lgf

#endif