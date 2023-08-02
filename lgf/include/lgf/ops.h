#ifndef INTERNALOPS_H_
#define INTERNALOPS_H_
#include "lgf/operation.h"
#include "lgf/LGFContext.h"
#include "lgf/group.h"
#include "types.h"
#include <string>
namespace lgf
{

class moduleOp : public graph{
    public:
    moduleOp() : graph("module"){}
    ~moduleOp(){}
    static moduleOp * build(LGFContext *ctx){
        auto op = new moduleOp();
        return op;
    }
    virtual std::string represent() {return getSID();}
};

class declOp : public operation{
    public:
    declOp() : operation("declOp") {}
    static declOp * build(LGFContext *ctx, type_t type) {
        auto op = new declOp();
        op->setSID("declOp");
        op->createValue(type, "");
        return op;
    }
    value * output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" : Declare";
        return p.dump();
    }
};

class assignOp : public operation{
    public:
    assignOp() : operation("assign") {}
    ~assignOp() { }
    static assignOp * build(LGFContext *ctx, value* lhs, value* rhs){
        auto op = new assignOp();
        op->createValue(rhs->getType(), "");
        op->registerInput(lhs, rhs);
        return op;
    }
    value * lhs() { return inputValue(0); }
    value * rhs() { return inputValue(1); }
    // note the outputValue(0) is the dependecyValue;
    value * output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(1)->represent()<<" -> "<<inputValue(0)->represent();
        return p.dump();
    }
};

//----------------------------------------

class cstDeclOp : public lgf::operation {
    public:
    cstDeclOp () = default;
    static cstDeclOp *build(LGFContext *ctx, int val_){
        auto op = new cstDeclOp();
        op->intValue = val_;
        op->isInt = 1;
        op->createValue(ctx->getType<intType>(), "");
        return op;
    }
    static cstDeclOp *build(LGFContext *ctx, double val_){
        auto op = new cstDeclOp();
        op->doubleValue = val_;
        op->createValue(ctx->getType<doubleType>(), "");
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

class funcDefineOp : public graph {
    public:
    funcDefineOp() : graph("funcDefineOp") {}
    static funcDefineOp* build(LGFContext *ctx, std::string id_, lgf::type_t returnType_){
        auto op = new funcDefineOp();
        op->returnType = returnType_;
        op->id = id_;
        op->createValue(ctx->getType<mapping_t>(),"");
        return op;
    }
    // this builder for no return type func defining
    static funcDefineOp* build(LGFContext *ctx, std::string id_){
        auto op = new funcDefineOp();
        op->id = id_;
        return op;
    }
    void registerArg(type_t type, std::string id){
        getEntry().createValue(type, id);
    }
    std::string id;
    virtual std::string represent(){ 
        printer p;
        p<<"funcOp: "<<id<<" (";
        p<<getEntry().representOutputs()<<")";
        if(returnType.getImpl()) p<<" -> "<<returnType.represent(); 
        return p.dump();
    }
    lgf::type_t returnType;
};

class funcCallOp : public operation{
    public:
    funcCallOp() = default;
    static funcCallOp * build(LGFContext *ctx, value* callee){
        auto op = new funcCallOp();
        op->registerInput(callee);
        auto& ret = callee->getDefiningOp<funcDefineOp>()->returnType;
        if(ret.getImpl()){
            op->hasReturn = 1;
            op->createValue(ret, "");
        }
        return op;
    }
    template<typename ...ARGS>
    static funcCallOp * build(LGFContext *ctx, value* callee, ARGS ... args){
        auto op = build(ctx, callee);
        op->registerInput(args...);
        return op;
    }
    value * getCallee(){ return inputValue(0); }
    value * arg(int n=0 ){ return inputValue(1); }
    value * returnValue() { return outputValue(1); }
    virtual std::string represent(){
        printer p;
        if( hasReturn ) p<<representOutputs()<<" = ";
        p<<"func callOp: "<<returnValue()->getDefiningOp<funcDefineOp>()->id<<" ("<<representInputs()<<")";
    }
    bool hasReturn = 0;
};

class returnOp : public operation {
    public:
    returnOp() = default;
    static returnOp * build(LGFContext *ctx){
        auto op = new returnOp();
        return op;
    }
    static returnOp * build(LGFContext *ctx, value* val){
        auto op = build(ctx);
        op->registerInput(val);
        return op;
    }
    virtual std::string represent(){return "return";}
};

}
#endif