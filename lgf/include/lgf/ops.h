#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{

class moduleOp : public graph{
public:
    moduleOp() : graph("module"){}
    ~moduleOp(){}
    static moduleOp * build(){
        auto op = new moduleOp();
        return op;
    }
    virtual std::string represent() {return getSID();}
};

class declOp : public operation{
    public:
    declOp() = default;
    static declOp * build(type_t type) {
        auto op = new declOp();
        op->setSID("declOp");
        op->createValue(type, "");
        return op;
    }
    value * output(){ return &outputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID();
        return p.dump();
    }
};

class assignOp : public operation{
    public:
    assignOp() = default;
    ~assignOp() { }
    static assignOp * build(type_t type, value lhs, value rhs){
        auto op = new assignOp();
        op->setSID("assignOp");
        op->createValue(type, "");
        op->registerInput(lhs, rhs);
        return op;
    }
    value * lhs() { return &inputValue(0); }
    value * rhs() { return &inputValue(1); }
    value * output(){ return &outputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(1).represent()<<" -> "<<inputValue(0).represent();
        return p.dump();
    }
};
//----------------------------------------

}
#endif