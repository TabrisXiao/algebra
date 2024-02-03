#ifndef MATH_SIO_OPS_H
#define MATH_SIO_OPS_H
#include "lgf/operation.h"
#include "libs/builtin/types.h"
#include "libs/AAB/types.h"
#include "lgf/group.h"

namespace lgf::SIO{

class representOp : public operation{
    public:
    representOp(std::string name) : operation(name) {}
    void addInputs(const std::vector<value *> &args){
        for(auto & val : args){
            registerInput(val);
        }
    }
    void addOutput(type_t type, std::string sid){
        createValue(type, sid);
    }
};

class symbolOp: public representOp{
    public:
    symbolOp() : representOp("sio::symbol") {}
    static symbolOp* build(lgf::LGFContext, type_t type, std::string sid){
        symbolOp *op = new symbolOp();
        op->addOutput(type, sid);
        return op;
    }
    std::string getSymbol(){
        return outputValue(1)->getSID();
    }
    virtual std::string represent() override{
        printer p;
        p<<representOutputs()<<" = "<<getSID();
        return p.dump();
    }
};

class sumOp : public representOp{
    public:
    sumOp() : representOp("sio::sum") {}
    static sumOp* build(lgf::LGFContext, type_t type, std::string sid, const std::vector<value *> args){
        sumOp *op = new sumOp();
        op->addInputs(args);
        op->addOutput(type, sid);
        return op;
    }
};

class scalarProductOp : public representOp{
    public:
    scalarProductOp() : representOp("sio::scalarProduct") {}
    static scalarProductOp* build(lgf::LGFContext, type_t type, std::string sid, const std::vector<value *> args){
        scalarProductOp *op = new scalarProductOp();
        op->addInputs(args);
        op->addOutput(type, sid);
        return op;
    }
};

class equalOp : public representOp{
    public:
    equalOp() : representOp("sio::equal") {}
    static equalOp* build(lgf::LGFContext, value* lhs, value* rhs){
        equalOp *op = new equalOp();
        op->registerInput(lhs, rhs);
        return op;
    }
};

class assignOp : public representOp{
    public:
    assignOp() : representOp("sio::assign") {}
    static assignOp* build(lgf::LGFContext, value* lhs, value* rhs){
        assignOp *op = new assignOp();
        op->registerInput(lhs, rhs);
        op->createValue(lhs->getType(), lhs->getSID());
        return op;
    }
};

class funcOp : public representOp{
    public:
    funcOp() : representOp("sio::func") {}
    static funcOp* build(lgf::LGFContext, type_t type, std::string sid, const std::vector<value *> args){
        funcOp *op = new funcOp();
        op->addInputs(args);
        op->addOutput(type, sid);
        return op;
    }
    void setFuncName(std::string name){
        funcName = name;
    }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<" sio::func "<<funcName<<"( "<<representInputs()<<" )";
        return p.dump();
    }
    std::string getFuncName(){ return funcName; }
    std::string funcName;
};

}

#endif