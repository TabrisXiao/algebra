#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{
class subGraphDefBaseOp : public operation{
    public:
    subGraphDefBaseOp(std::string id) {setTypeID(id);};
    ~subGraphDefBaseOp() = default;
    virtual graph* getSubgraph() override final {return &block;}
    void assignID(int n=0){ block.assignID(n); }
    graph block;
};
class moduleOp : public subGraphDefBaseOp{
public:
    moduleOp() : subGraphDefBaseOp("module"){}
    ~moduleOp(){}
    std::string represent() {return getTypeID();}
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

class funcDefOp : public subGraphDefBaseOp {
public:
    funcDefOp(std::string funcid, std::vector<std::string> args) : subGraphDefBaseOp("funcDef") {
        funcID = funcid;
        argTypes = args;
    }
    std::string getFuncID(){return funcID;}
    std::string represent() {
        printer p;
        p<<funcID<<" = "<<getTypeID()<<"(";
        p<<argTypes[0];
        for(auto iter=argTypes.begin()+1; iter!=argTypes.end(); iter++){
            p<<", "<<(*iter);
        }
        p<<") ";
        return p.dump();
    }
    
    std::string funcID;
    std::vector<std::string> argTypes;
};
}
#endif