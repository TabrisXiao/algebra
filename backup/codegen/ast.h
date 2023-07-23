
#ifndef CODEGEN_AST_H
#define CODEGEN_AST_H

#include "lgf/lgf.h"
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
    static binaryExprAST* build(value& lhs, value& rhs, std::string op){
        auto opa = new binaryExprAST(lhs, rhs, op);
        return opa;
    }
    value& lhs() {return inputValue(0);}
    value& rhs() {return inputValue(1);}
    virtual std::string represent(){
            printer p;
        p<<outputValue().represent();
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
    static defFuncAST* build(std::string funcid){
        auto op = new defFuncAST(funcid);
        return op;
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
    type_t getReturnType(){return outputValue().getType();}
    value& arg(int n=0){ return getEntry().outputValue(n);}
    std::string funcID="Unknown";
};

class returnAST : public operation, public ASTBase{
    public: 
    returnAST() : operation("return"), ASTBase(ASTKind::Return) {};
    static returnAST* build(){
        auto op = new returnAST();
        return op;
    }
    static returnAST* build(value& val){
        auto op = new returnAST();
        op->registerInput(val);
        return op;
    }
    std::string represent()override{
        printer p;
        p<<getSID()<<" : "<<inputValue().represent();
        return p.dump();
    }
};
//----------------------------------------

class declValueAST : public operation, public ASTBase{
    public : 
    declValueAST() 
    : operation("ValueDecl")
    , ASTBase(ASTKind::DeclVar)
    { }
    static declValueAST* build(std::string type){
        auto op = new declValueAST();
        auto &val = op->createValue();
        val.setType(type);
        return op;
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
    template<typename ...ARGS>
    static callFuncAST* build(defFuncAST* op, ARGS ...args){
        auto op = new callFuncAST(op, args, ...);
        return op;
    }
    std::string represent() final {
        printer p;
        p<<representOutputs()<<" = "<<getSID()<<" : ";
        p<<funcID<<"(%"<<inputValue().represent();
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