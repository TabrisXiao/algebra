#include "unit_test_frame.h"
#include "lgf/node.h"
#include "lgf/painter.h"
#include <string>
#include <vector>

using namespace lgf;
namespace test_body
{

    class real : public valueDesc
    {
    public:
        real() = default;
        virtual sid_t represent_type() override
        {
            return "real";
        }
    };

    class defop : public node
    {
    public:
        defop(std::string name)
        {
            set_sid("def");
        }
        static defop *build(LGFContext *ctx, valueDesc *desc, std::string name)
        {
            auto op = new defop(name);
            op->set_value_desc(desc);
            return op;
        }
        virtual sid_t represent() override
        {
            printer p;
            p << value_rep();
            p << " = " << get_sid() << " : " << get_value_desc()->represent();
            return p.dump();
        }
    };
    class addop : public node
    {
    public:
        addop() = default;
        static addop *build(LGFContext *ctx, node *lhs, node *rhs)
        {
            auto op = new addop();
            op->set_sid("add");
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
            p << " = " << get_sid() << " : " << lhs()->get_value_sid() << " + " << rhs()->get_value_sid();
            return p.dump();
        }
    };

    class multiplyop : public node
    {
    public:
        multiplyop() = default;
        static multiplyop *build(LGFContext *ctx, node *lhs, node *rhs)
        {
            auto op = new multiplyop();
            op->set_sid("multiply");
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
            p << " = " << get_sid() << " : " << lhs()->get_value_sid() << " * " << rhs()->get_value_sid();
            return p.dump();
        }
    };

    class returnop : public node
    {
    public:
        returnop() = default;
        static returnop *build(LGFContext *ctx, node *inputValue)
        {
            auto op = new returnop();
            op->set_sid("return");
            op->register_input(inputValue);
            return op;
        }
        virtual sid_t represent() override
        {
            printer p;
            p << get_sid() << " " << input()->value_rep();
            return p.dump();
        }
    };

    class test_node : public test_wrapper
    {
    public:
        test_node() { test_id = "node test"; };
        bool run()
        {
            graph reg;
            bool test_result = 0;
            painter pntr;
            pntr.goto_graph(&reg);
            auto real_desc = new real();
            auto x = pntr.paint<defop>(real_desc, "x");
            auto y = pntr.paint<defop>(real_desc, "y");
            auto z = pntr.paint<defop>(real_desc, "z");
            auto xy = pntr.paint<addop>(x, y);
            auto ret = pntr.paint<returnop>(xy);
            reg.print();
            auto xy2 = pntr.replace_op<multiplyop>(xy, x, y);

            // special case: the op replaced is input of new one
            // auto xy3 = pntr.replace_op<multiplyop>(xy2, xy2,y);
            reg.clean();

            reg.print();
            return 0;
        }
    };
};
