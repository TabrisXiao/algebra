#ifndef MATH_SIO_OPS_H
#define MATH_SIO_OPS_H
#include "lgf/operation.h"
#include "libs/builtin/types.h"
#include "libs/AAB/types.h"
#include "lgf/group.h"
#include "pattern.h"

namespace lgf::SIO{

class representOp : public operation{
    public:
    representOp(std::string name) : operation(name) {}
    void addInputs(const std::vector<operation *> &args){
        for(auto & val : args){
            registerInput(val);
        }
    }
    void addOutput(type_t type, std::string sid){
        createValue(type, sid);
    }
};

class sumOp : public representOp{
    public:
    sumOp() : representOp("sio::sum") {}
    static sumOp* build(lgf::LGFContext, type_t type, std::string sid, const std::vector<operation *> args){
        sumOp *op = new sumOp();
        op->addInputs(args);
        op->addOutput(type, sid);
        return op;
    }
};

class scalarProductOp : public representOp{
    public:
    scalarProductOp() : representOp("sio::scalarProduct") {}
    static scalarProductOp* build(lgf::LGFContext, type_t type, std::string sid, const std::vector<operation *> args){
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

}

#endif