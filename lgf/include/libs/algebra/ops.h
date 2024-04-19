
#ifndef LGF_LIB_ALGEBRA_OPS_H
#define LGF_LIB_ALGEBRA_OPS_H

#include "lgf/group.h"
#include "desc.h"

namespace lgf{
    class sumOp : public node{
        public:
        sumOp() : node("sum") {}
        static sumOp* build(std::vector<node*>& vec){
            auto op = new sumOp();
            op->register_inputs(vec);
            op->infer_trivial_value_desc();
            return op;
        }
        template<typename... Args>
        static sumOp* build(Args... args){
            auto op = new sumOp();
            op->register_input(args...);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class productOp : public node, public normalizer {
        public:
        productOp() : node("product") {}
        static productOp* build(std::vector<node*>& vec){
            auto op = new productOp();
            op->register_inputs(vec);
            op->infer_trivial_value_desc();
            return op;
        }
        static productOp* build(){
            auto op = new productOp();
            return op;
        }
        template<typename... Args>
        static productOp* build(Args... args){
            auto op = new productOp();
            op->register_input(args...);
            op->infer_trivial_value_desc();
            return op;
        }
        virtual resultCode rewrite(painter& p, node* op) override;
    };

    class negativeOp : public node {
        public:
        negativeOp() : node("negative") {}
        static negativeOp* build(node* n){
            auto op = new negativeOp();
            op->register_input(n);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class inverseOp : public node{
        public:
        inverseOp() : node("inverse") {}
        static inverseOp* build(node* n){
            auto op = new inverseOp();
            op->register_input(n);
            op->infer_trivial_value_desc();
            return op;
        }
    };

    class minusOp : public node {
        public:
        minusOp() : node("minus") {}
        static minusOp* build(node* lhs, node* rhs){
            auto op = new minusOp();
            op->register_input(lhs, rhs);
            op->infer_trivial_value_desc();
            return op;
        }
        node* lhs() {return input(0);}
        node* rhs() {return input(1);}
        virtual sid_t represent() override{
            printer p;
            p<<value_rep();
            p<<" = "<<get_sid()<<" : "<<lhs()->get_sid() << " - "<<rhs()->get_sid();
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