
#ifndef LGF_FUNCTIONAL_ANALYSIS_OPS_H
#define LGF_FUNCTIONAL_ANALYSIS_OPS_H
#include "libs/aab/ops.h"
#include "libs/fa/types.h"

namespace lgf
{

    class funcSineOp
    {
    public:
        funcSineOp() : mappingOp("sine") {}
        static funcSineOp *build( node *x)
        {
            auto op = new funcSineOp();
            op->register_input(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcCosOp
    {
    public:
        funcCosOp() : mappingOp("cos") {}
        static funcCosOp *build( node *x)
        {
            auto op = new funcCosOp();
            op->register_input(x);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class funcPowerOp
    {
    public:
        powerOp() : mappingOp("power") {}
        static funcPowerOp *build( node* x, double n)
        {
            auto op = new powerOp();
            op->register_input(x);
            op->setPower(n);
            op->infer_trivial_value_desc();
            return op;
        }
        void setPower(double n) { p = n; }
        double power() { return p; }
        double p = 1;
    };

    class funcExpOp
    {
    public:
        funcExpOp() : mappingOp("functional::exp") {}
        static expOp *build(LGFContext *ctx, node *power)
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

    class partialDifferentiateOp
    {
    public:
        partialDifferentiateOp() : mappingOp("PartialDifferentiate") {}
        static partialDifferentiateOp *build( node *func, node *var)
        {
            auto op = new partialDifferentiateOp();
            op->register_input(func, var);
            op->set_value_desc(func->get_value_desc());
            return op;
        }
        node *func() { return input(0); }
        node *var() { return input(1); }
    };

    class differentiateOp
    {
    public:
        differentiateOp() : mappingOp("differentiate") {}
        static differentiateOp *build(LGFContext *ctx, node *input, node *target)
        {
            auto op = new differentiateOp();
            op->register_input(input, target);
            op->set_value_desc(input->get_value_desc());
            return op;
        }
        node *input() { return input(0); }
        node *target() { return input(1); }
    };

    class unionOp : public operation
    {
    public:
        unionOp() : operation("functional::union") {}
        template <typename... ARGS>
        static unionOp *build(LGFContext *ctx, ARGS... args)
        {
            auto op = new unionOp();
            op->registerInput(args...);
            op->createnode(op->inputnode()->getType());
            return op;
        }
    };

    class intersectOp : public operation
    {
    public:
        intersectOp() : operation("functional::intersect") {}
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