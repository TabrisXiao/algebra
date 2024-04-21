
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
        edgeBundle &get_args()
        {
            return get_input_handles();
        }

    private:
        size_t narg = 0;
    };

    class funcSineOp : public mappingOp
    {
    public:
        funcSineOp() : mappingOp("sine") {}
        static funcSineOp *build(LGFContext *ctx, node *x)
        {
            auto op = new funcSineOp();
            op->add_args(x);
            op->infer_trivial_value_desc();
            return op;
        }
        virtual sid_t represent() override
        {
            return get_value_sid() + " = sin( " + input()->get_value_sid() + " )";
        }
    };

    class funcCosOp : public mappingOp
    {
    public:
        funcCosOp() : mappingOp("cos") {}
        static funcCosOp *build(LGFContext *ctx, node *x)
        {
            auto op = new funcCosOp();
            op->add_args(x);
            op->infer_trivial_value_desc();
            return op;
        }
        virtual sid_t represent() override
        {
            return value_rep() + " = cos( " + input()->get_value_sid() + " )";
        }
    };

    class funcPowerOp : public mappingOp
    {
    public:
        funcPowerOp() : mappingOp("power") {}
        static funcPowerOp *build(LGFContext *ctx, node *x, double n)
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
        virtual sid_t represent() override
        {
            return value_rep() + " = " + input()->get_value_sid() + "power=" + std::to_string(p);
        }
    };

    class funcExpOp : public mappingOp
    {
    public:
        funcExpOp() : mappingOp("functional::exp") {}
        static funcExpOp *build(LGFContext *ctx, node *power)
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
        virtual sid_t represent() override
        {
            return value_rep() + " = exp( " + input()->get_value_sid() + " )";
        }
    };

    class partialDifferentiateOp : public mappingOp
    {
    public:
        partialDifferentiateOp() : mappingOp("PartialDifferential") {}
        static partialDifferentiateOp *build(LGFContext *ctx, node *func, node *var, int order = 1)
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
            op->add_args(input);
            op->set_value_desc(input->get_value_desc());
            return op;
        }
        virtual sid_t represent() override
        {
            auto res = value_rep() + " = " + get_sid() + ": " + input()->get_value_sid();
            return res;
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
            op->add_args(args...);
            op->set_value_desc(op->input()->get_value_desc());
            return op;
        }
    };

} // namespace lgf

#endif