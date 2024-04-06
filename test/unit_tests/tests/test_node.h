#include "unit_test_frame.h"
#include "lgf/node.h"
#include "lgf/painter.h"
#include <string>
#include <vector>

using namespace lgf;
namespace test_body{

class real : public valueDesc {
    public:
    real() = default;
    virtual sid_t represent() override{
        return "real";
    }
};

class defop : public node {
    public:
    defop (std::string name){
        set_sid("def");
    }
    static defop* build( valueDesc* desc, std::string name){
        auto op = new defop(name);
        op->output()->set_desc(desc);
        return op;
    }
    virtual sid_t represent() override{
        printer p;
        p<<represent_output();
        p<< " = "<<get_sid()<<" : "<<output()->get_desc()->represent();
        return p.dump();
    }
};
class addop : public node {
    public:
    addop () = default;
    static addop* build( node* lhs, node* rhs){
        auto op = new addop();
        op->accept(lhs, rhs);
        op->output()->set_desc(lhs->output()->get_desc());
        return op;
    }
    value* lhs() {return input(0);}
    value* rhs() {return input(1);}
    virtual sid_t represent() override{
        printer p;
        p<<represent_output();
        p<<" = "<<get_sid()<<" : "<<lhs()->get_sid() << " + "<<rhs()->get_sid();
        return p.dump();
    }
};

class multiplyop : public node {
    public:
    multiplyop () = default;
    static multiplyop* build( node * lhs, node * rhs){
        auto op = new multiplyop();
        op->accept(lhs, rhs);
        op->output()->set_desc(lhs->output()->get_desc());
        op->set_sid("multiply");
        return op;
    }
    value* lhs() {return input(0);}
    value* rhs() {return input(1);}
    virtual sid_t represent() override{
        printer p;
        p<<represent_output();
        p<<" = "<<get_sid()<<" : "<<lhs()->get_sid() << " * "<<rhs()->get_sid();
        return p.dump();
    }
};

class returnop : public node{
    public: 
    returnop () = default;
    static returnop * build( node * inputValue){
        auto op = new returnop();
        op->accept(inputValue);
        op->set_sid("return");
        return op;
    }
    virtual sid_t represent() override{
        printer p;
        p<<get_sid()<<" "<<input()->represent();
        return p.dump();
    }
};

class test_node : public test_wrapper{
    public:
    test_node() {test_id = "node test";};
    bool run() {
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
        auto xy2 = pntr.replace_op<multiplyop>(xy, x,y);
        reg.clean();
        xy2->replace_input(1, z);

        reg.print();
        return 0;
    }
};
};
