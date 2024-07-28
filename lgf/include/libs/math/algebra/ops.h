
#ifndef LGF_LIB_ALGEBRA_OPS_H
#define LGF_LIB_ALGEBRA_OPS_H

#include "libs/builtin/builtin.h"
#include "lgf/context.h"
#include "desc.h"

namespace lgf
{
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

    // class commuteOp : public node, public graphOperation{
    //     public:
    //     commuteOp() : node("commute") {}
    //     static commuteOp* build(node* lhs, node* rhs){
    //         auto op = new commuteOp();
    //         op->register_input(lhs, rhs);
    //         op->infer_trivial_value_desc();
    //         return op;
    //     }
    //     node* lhs() {return input(0);}
    //     node* rhs() {return input(1);}
    //     virtual sid_t represent() override{
    //         printer p;
    //         p<<value_rep();
    //         p<<" = "<<get_sid()<<" : ["<<lhs()->get_sid() << ", "<<rhs()->get_sid()<<"]";
    //         return p.dump();
    //     }
    //     virtual resultCode rewrite(painter p, node* op) override{
    //         auto lhs = op->input(0);
    //         auto rhs = op->input(1);
    //         p->replace_node(op, commuteOp::build(rhs, lhs));
    //         return resultCode::rewrite();
    //     }
    // };
}

#endif