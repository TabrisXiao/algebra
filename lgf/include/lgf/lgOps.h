#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{
class moduleOp : public operation{
public:
    moduleOp(){ setTypeID("module"); }
    ~moduleOp(){std::cout<<"deleted"<<std::endl;}
    virtual graph* getSubgraph() override final {return &block;}
    std::string represent() {return getTypeID();}
    virtual void printOp() override final{
        global::stream::getInstance().printIndent();
        global::stream::getInstance()<<getTypeID()<<" : ";
        block.print();
        global::stream::getInstance()<<"\n";
    }
    void assignID(int n=0){ block.assignID(n); }
    graph block;
};

// a binary op is an operation taking exactly two values as inputs
// and produce eactly one output value
class binaryOp : public operation{
    public:
    binaryOp(value &lhs, value &rhs){
        registInput(lhs, rhs);
        createValue();
    }
    value& lhs() {return input(0);}
    value& rhs() {return input(1);}
};

// class funcOp : public operation {
// public:
//     funcOp(std::string funcid){
//         funcID = funcid;
//     }
//     std::string getFuncID(){return funcID;}

//     std::string funcID;
// };
}
#endif