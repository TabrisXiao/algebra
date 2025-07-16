#ifndef LIB_QFT_QFT_H_
#define LIB_QFT_QFT_H_
#include "graph/operation.h"

namespace qft
{
    using namespace lgf;

    class Operator : public descBase
    {
    public:
        Operator(const char *name = "qft::operator") : descBase(name) {}
        static description get()
        {
            return description::get<Operator>();
        }
    };

    class State : public descBase
    {
    public:
        State(const char *name = "qft::state") : descBase(name) {}
        static description get()
        {
            return description::get<State>();
        }
    };

    class Commutator : public operation
    {
    public:
        Commutator(region *r = nullptr) : operation("commutator", r) {}
        static std::unique_ptr<Commutator> build(region *r, object *input1, object *input2)
        {
            auto op = std::make_unique<Commutator>(r);
            op->create_object(description::get<Operator>());
            op->register_input(input1);
            op->register_input(input2);
            return std::move(op);
        }
    };

    class Conjugate : public operation
    {
    public:
        Conjugate(region *r = nullptr) : operation("conjugate", r) {}
        static std::unique_ptr<Conjugate> build(region *r, object *input)
        {
            auto op = std::make_unique<Conjugate>(r);
            op->create_object(input->get_desc());
            op->register_input(input);
            return std::move(op);
        }
    };
}

#endif