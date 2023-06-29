#include "unit_test_frame.h"
#include "lgf/operation.h"
#include "lgf/painter.h"
#include "lgf/lgf.h"
#include <string>

using namespace lgf;
namespace test_body{
class defop : public operation {
    public:
    defop (std::string name){
        setSID("def");
        auto& v = createValue();
        v.setSID(name);
    }
    static defop* build(std::string name){
        auto op = new defop(name);
        return op;
    }
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<< " = "<<getSID();
        return p.dump();
    }
};
class addop : public operation {
    public:
    addop () = default;
    static addop* build(value& lhs, value& rhs){
        auto op = new addop();
        op->registerInput(lhs, rhs);
        op->setSID("add");
        op->createValue();
        return op;
    }
    value& lhs() {return inputValue(0);}
    value& rhs() {return inputValue(1);}
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : "<<lhs().represent() << " + "<<rhs().represent();
        return p.dump();
    }
};

class multiplyop : public operation {
    public:
    multiplyop () = default;
    static multiplyop* build(value& lhs, value& rhs){
        auto op = new multiplyop();
        op->registerInput(lhs, rhs);
        op->setSID("multiply");
        op->createValue();
        return op;
    }
    value& lhs() {return inputValue(0);}
    value& rhs() {return inputValue(1);}
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : "<<lhs().represent() << " * "<<rhs().represent();
        return p.dump();
    }
};

class returnop : public operation{
    public: 
    returnop () = default;
    static returnop * build(value & inputValue){
        auto op = new returnop();
        op->registerInput(inputValue);
        op->setSID("return");
        op->createValue();
        return op;
    }
    virtual std::string represent() override{
        printer p;
        p<<outputValue().represent();
        p<<" = "<<getSID()<<" : "<<inputValue().represent();
        return p.dump();
    }
};

class test_operation : public test_wrapper{
    
    public:
    test_operation() {test_id = "operation test";};
    bool run() {
        canvas reg;
        bool test_result = 0;
        painter pntr(&reg);
        auto x = pntr.createOp<defop>("x");
        auto y = pntr.createOp<defop>("y");
        auto z = pntr.createOp<defop>("z");
        auto xy = pntr.createOp<addop>(x->outputValue(), y->outputValue());
        auto ret = pntr.createOp<returnop>(xy->outputValue());
        reg.assignID();
        reg.print();
        auto xy2 = pntr.replaceOp<multiplyop>(xy, x->outputValue(),y->outputValue());
        reg.clean();
        xy2->replaceInputValue(1, z->outputValue());
        reg.assignID();

        reg.print();
        return 0;
    }
};
};
