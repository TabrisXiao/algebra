
#ifndef CODEGEN_AST_H
#define CODEGEN_AST_H

#include "lgf/operation.h"
#include "lgf/lgOps.h"
#include "lgf/printer.h"

namespace lgf{

class defClass : public subGraphDefBaseOp {
    public:
    defClass(std::string id ) : subGraphDefBaseOp(id) {}
    std::string represent() {
        printer p;
        p<<"#"<<getTypeID()<<" = ClassDef";
        return p.dump();
    }
};
class defFunc : public subGraphDefBaseOp {
    public:
    defFunc(std::string id, std::string type) : subGraphDefBaseOp(id) {
        returnType = type;
    }
    std::string represent() {
        printer p;
        p<<"#"<<getTypeID()<<" = FuncDef";
        return p.dump();
    }
    std::string returnType;
};
class callFunc : public operation {
    public:
    callFunc(std::string id) {
        setTypeID("FuncCall");
        auto &val = createValue();
    }
};
class declVar : public operation{
    public : 
    declVar(std::string type) 
    { 
        setTypeID("declVar"); 
        auto &val = createValue();
        val.setTypeID(type);
    }
    std::string represent() final {
        printer p;
        p<<representOutputs()<<getTypeID();
        return p.dump();
    }
};
}

#endif