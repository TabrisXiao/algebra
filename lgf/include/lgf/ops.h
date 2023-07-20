#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
#include "types.h"
#include <string>
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
        p<<representOutputs()<<" : Declare";
        return p.dump();
    }
};

class assignOp : public operation{
    public:
    assignOp() = default;
    ~assignOp() { }
    static assignOp * build(type_t type, value lhs, value rhs){
        auto op = new assignOp();
        op->setSID("assign");
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

class cstDeclOp : public lgf::operation {
    public:
    cstDeclOp () = default;
    static cstDeclOp *build(int val_){
        auto op = new cstDeclOp();
        op->intValue = val_;
        op->isInt = 1;
        op->createValue(intType::build(), "");
        return op;
    }
    static cstDeclOp *build(double val_){
        auto op = new cstDeclOp();
        op->doubleValue = val_;
        op->createValue(doubleType::build(), "");
        return op;
    }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<"Constant ";
        auto & v = isInt ? "int: "+std::to_string (intValue) : "double: "+std::to_string (doubleValue);
        p<<v;
        return p.dump();
    }
    bool isInt = 0;
    int intValue;
    double doubleValue;
};

}
#endif