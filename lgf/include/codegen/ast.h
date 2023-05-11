
#ifndef CODEGEN_AST_H
#define CODEGEN_AST_H

#include "lgf/operation.h"
#include "lgf/lgOps.h"
#include "lgf/printer.h"

namespace lgf{
namespace ast{

enum ASTKind{
    DefFunc,
    CallFunc,
    DeclVar,
    ExprVar,
    ExprNum,
    ExprLiteral,
    ExprBinary,
    Return,
    Unknown
};

class ASTBase {
    public:
    ASTBase(ASTKind k) : kind(k){}
    virtual ~ASTBase() {};
    ASTKind getKind(){ return kind; }
    bool checkKind(ASTKind &k){ return kind == k; }
    ASTKind kind = ASTKind::Unknown;
};

class defFuncAST : public graph, public ASTBase{
    public:
    template<typename ...ARGS>
    defFuncAST(std::string funcid, std::string ty, ARGS ... args)
    :graph("FuncDef")
    ,ASTBase(ASTKind::DefFunc)
    ,returnType(ty)
    {
        funcID = funcid;
        (registerInputType(args),...);
    }
    void registerInputType(std::string ty){
        auto &val = getEntry().createValue();
        val.setTypeID(ty);
        val.setSID("arg");
    }
    virtual std::string represent(){
        printer p;
        p<<"@"<<funcID<<" "<<returnType.represent()<<" = "<<getSID()<<" (";
        p<<getEntry().representOutputs()<<")";
        return p.dump();
    }
    type_t getReturnType(){return returnType;}
    value& arg(int n=0){ return getEntry().output(n);}
    std::string funcID="Unknown";
    type_t returnType;
};

class returnAST : public operation, public ASTBase{
    public: 
    returnAST() : operation("return"), ASTBase(ASTKind::Return) {};
    returnAST(value& val): operation("return"), ASTBase(ASTKind::Return){
        registerInput(val);
    }
    std::string represent()override{
        printer p;
        p<<getSID()<<" : "<<input().represent();
        return p.dump();
    }
};
//----------------------------------------

class declValueAST : public operation, public ASTBase{
    public : 
    declValueAST(std::string type) 
    : operation("ValueDecl")
    , ASTBase(ASTKind::DeclVar)
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

class callFuncAST : public operation, public ASTBase {
    public:
    template<typename ...ARGS>
    callFuncAST(defFuncAST* op, ARGS ...args)
    : operation("Func")
    , ASTBase(ASTKind::CallFunc) {
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


} // namespace ast
} // namespace lgf

#endif