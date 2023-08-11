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
    static moduleOp * build(LGFContext *ctx, std::string id = ""){
        auto op = new moduleOp();
        op->name = id;
        op->createValue(ctx->getType<reference_t>(),"");
        return op;
    }
    value* output(){ return outputValue(1);}
    std::string name="";
    virtual std::string represent() {return getSID()+" "+name;}
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

class referenceOp : public operation {
    public:
    referenceOp() : operation ("ref") {}
    ~referenceOp() {}
    static referenceOp * build(LGFContext* ctx, type_t retType, value* val){
        auto op = new referenceOp();
        op->createValue(retType, "");
        op->refValue = val;
        return op; 
    }
    bool isRefValid(){
        if(!refValue) return 0;
        if(auto ptr = refValue->getDefiningOp<operation>()) return 1;
        refValue = nullptr;
        return refValue;
    }
    value* output(){ return outputValue(1);}
    value * refValue = nullptr;
    virtual std::string represent(){
        return representOutputs()+" = @"+refValue->represent();
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
    value* output(){ return outputValue(1); }
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
        op->createValue(ctx->getType<reference_t>(),"");
        return op;
    }
    // this builder for no return type func defining
    static funcDefineOp* build(LGFContext *ctx, std::string id_){
        auto op = new funcDefineOp();
        op->id = id_;
        op->createValue(ctx->getType<reference_t>(),"");
        return op;
    }
    void registerArg(type_t type, std::string id){
        getEntry().createValue(type, id);
    }
    value* getCallee(){ return outputValue(1); }
    value* argument(int n) { return getEntry().outputValue(n+1); }
    std::string id;
    virtual std::string represent(){ 
        printer p;
        p<<representOutputs()<<" = func ";
        if(isAbstract)p<<"Register";
        else p<<"Def";
        p<<" : "<<id<<" (";
        p<<getEntry().representOutputs()<<")";
        if(returnType.getImpl()) p<<" -> "<<returnType.represent(); 
        return p.dump();
    }
    bool isAbstract = 1;
    lgf::type_t returnType;
    virtual void print(){
        global::stream::getInstance().printIndent();
        std::string code = represent();
        // add space if the represent is not empty
        // {} no reprsent, shoudn't have space
        // module {}, have represent "module", should have space
        // between "module" and the {}.
        global::stream::getInstance()<<represent();
        if(!isAbstract){
            printGraph();
        } else global::stream::getInstance()<<"\n";
    }
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
    void addArg(value* arg) {
        registerInput(arg);
    }
    void addArgs(std::vector<value*> & vec){
        for(auto arg : vec){
            registerInput(arg);
        }
    }
    value * getCallee(){ return inputValue(0); }
    value * arg(int n=0 ){ return inputValue(n+1); }
    value * returnValue() { return outputValue(1); }
    virtual std::string represent(){
        printer p;
        auto callee = getCallee()->getDefiningOp<funcDefineOp>();
        if( hasReturn ) p<<representOutputs()<<" = ";
        p<<"func call: %"<<callee->getCallee()->getTraceID()<<" "<<callee->id<<" (";
        if( getInputSize()>1){
            p<<arg(0)->represent();
            for(auto i = 1; i< getInputSize()-1; /* the first element is callee */ i++){
                p<<", "<<arg(i)->represent();
            }
        }
        p<<")";
        return p.dump();
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
        if(val) op->registerInput(val);
        return op;
    }
    virtual std::string represent(){
        std::string res = "return";
        if(getInputSize()) {
            res =res+" "+inputValue(0)->represent();
        }
        return res;
    }
};

}
#endif