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

class defFuncOp : public graph {
    public:
    template<typename ...ARGS>
    defFuncOp(std::string funcid, std::string ty, ARGS ... args)
    :graph("DefFunc")
    ,returnType(ty)
    {
        (registerInputType(args),...);
    }
    void registerInputType(std::string ty){
        auto &val = getEntry().createValue();
        val.setTypeID(ty);
        val.setSID("arg");
    }
    virtual std::string represent(){
        printer p;
        p<<"@"<<getSID()<<" "<<returnType.represent()<<" = FuncDef (";
        p<<getEntry().representOutputs()<<")";
        return p.dump();
    }
    type_t getReturnType(){return returnType;}
    value& arg(int n=0){ return getEntry().output(n);}
    std::string funcID="Unknown";
    type_t returnType;
};

class returnOp : public operation{
    public: 
    returnOp() : operation("return"){};
    returnOp(value& val):operation("return"){
        registerInput(val);
    }
    std::string represent()override{
        printer p;
        p<<getSID()<<" : "<<input().represent();
        return p.dump();
    }
};
//----------------------------------------

class declValue : public operation{
    public : 
    declValue(std::string type) : operation("DeclValue")
    { 
        auto &val = createValue();
        val.setType(type);
    }
    std::string represent() final {
        printer p;
        p<<representOutputs()<<" = "<<getSID();
        return p.dump();
    }
};
//----------------------------------------

class callFuncOp : public operation {
    public:
    template<typename ...ARGS>
    callFuncOp(defFuncOp* op, ARGS ...args)
    : operation("Func") {
        funcID = op->funcID;
        auto &val = createValue();
        val.setType(op->getReturnType());
    }
    std::string represent() final {
        printer p;
        p<<representOutputs()<<" = "<<getSID()<<" : ";
        p<<funcID<<"(%"<<input().represent();
        for(auto iter=getInputRefs().begin()+1; iter!=getInputRefs().end(); iter++)
        {
            p<<", %"<<(*iter).represent();
        }
        p<<")";
        return p.dump();
    }
    std::string funcID;
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