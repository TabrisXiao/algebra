
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

class binaryExprAST : public operation, public ASTBase {
    public:
    binaryExprAST(value& lhs, value& rhs, std::string op) : operation("binary"), ASTBase(ASTKind::ExprBinary) {
        bop = op;
        registerInput(lhs, rhs);
        auto &val = createValue();
        val.setType(lhs.getType());
    };
    value& lhs() {return input(0);}
    value& rhs() {return input(1);}
    virtual std::string represent(){
            printer p;
        p<<output().represent();
        p<<" = "<<getSID()<<" : "<<lhs().represent() << " "<<bop<<" "<<rhs().represent();
        return p.dump();
    }
    std::string bop = "";
};

class defFuncAST : public graph, public ASTBase{
    public:
    defFuncAST(std::string funcid)
    : graph("FuncDef")
    ,ASTBase(ASTKind::DefFunc) 
    {
        funcID = funcid;
    }
    void registerReturnType(std::string tp){
        createValue().setTypeID(tp);
    }
    template<typename ...ARGS>
    void registerInputTypes(ARGS ... args)
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
        p<<"@"<<funcID<<" ";
        if(getOutputSize() > 0){
            p<<getReturnType().represent();
        } 
        p<<" = "<<getSID()<<" (";
        p<<getEntry().representOutputs()<<")";
        return p.dump();
    }
    int nArgs(){return int(getEntry().getOutputSize()); }
    type_t getReturnType(){return output().getType();}
    value& arg(int n=0){ return getEntry().output(n);}
    std::string funcID="Unknown";
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