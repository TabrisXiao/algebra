#ifndef CODEGEN_CPPEWRITER_H
#define CODEGEN_CPPEWRITER_H

#include "stream.h"
#include "codeWriter.h"
#include "ast.h"

namespace codegen{
class cppTranslationRule : public translateRule<lgf::ast::ASTBase>{
    public:
    cppTranslationRule(): translateRule<lgf::ast::ASTBase>(){}
    void writeReturnAST(cgstream &out, lgf::operation *op);
    void writeDeclVarAST(cgstream &out, lgf::operation *op);
    void writeDefFuncAST(cgstream &out, lgf::operation *op);
    void writeBinaryAST(cgstream &out, lgf::operation *op);
    void writeFuncCallAST(cgstream &out, lgf::operation *op);
    virtual bool write(cgstream &out, lgf::ast::ASTBase *op_){
        auto op = dynamic_cast<lgf::operation*>(op_);
        switch(op_->getKind()){
            case lgf::ast::ASTKind::Return:
                writeReturnAST(out, op);
                return 1;
            case lgf::ast::ASTKind::DeclVar:
                writeDeclVarAST(out, op);
                return 1;
            case lgf::ast::ASTKind::DefFunc:
                writeDefFuncAST(out, op);
                return 1;
            case lgf::ast::ASTKind::ExprBinary:
                writeBinaryAST(out, op);
                return 1;
            default:
                return 0;
        }
        return 0;
    }
    virtual void enterGraphRule(cgstream &out, lgf::graph* reg) override{
        out<<"{\n";      
        out.incrIndentLevel();
    }
    virtual void exitGraphRule(cgstream &out, lgf::graph* reg) override {
        out.decrIndentLevel(); 
        out.printIndent(); 
        out<<"}\n";
    }
    // get the variable symbolicID
    std::string symbolicID(lgf::value &v){
        printer p;
        std::string sid = v.getSID();
        if(sid=="") sid = "x";
        p<<sid<<v.getTraceID();
        return p.dump();
    }
    std::string declVar(lgf::value &v){
        printer p;
        std::string sid = v.getSID();
        if(sid=="") sid = "x";
        p<<v.getType().getSID()<<" "<<sid<<v.getTraceID();
        return p.dump();
    }
};
}

#endif