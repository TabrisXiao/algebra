#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{
class canvas : public graph{
    public : 
    canvas() : graph("canvas"){}
    virtual std::string represent(){ return "";} 
};
//----------------------------------------

class moduleOp : public graph{
public:
    moduleOp() : graph("module"){}
    ~moduleOp(){}
    virtual std::string represent() {return getSID();}
};
//----------------------------------------

// a binary op is an operation taking exactly two values as inputs
// and produce eactly one output value
class binaryOp : public operation{
    public:
    binaryOp(value &lhs, value &rhs){
        registerInput(lhs, rhs);
        createValue();
    }
    value& lhs() {return input(0);}
    value& rhs() {return input(1);}
};
}
#endif