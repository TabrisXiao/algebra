#include "codegen/cppWriter.h"


namespace codegen{
void cppTranslationRule::writeReturnAST(cgstream &out, lgf::operation *op){
    out.printIndent();
    out<<"return";
    if(op->getInputSize()>0) out<<" x"<<op->input().getTraceID();
    out<<";\n";
}

void cppTranslationRule::writeDeclVarAST(cgstream &out, lgf::operation *op){
    out.printIndent();
    auto& var = op->output();
    out<<var.getType().getSID()<<" x"<<var.getTraceID()<<";\n";
}

void cppTranslationRule::writeDefFuncAST(cgstream &out, lgf::operation *op){
    out.printIndent();
    auto fop = dynamic_cast< lgf::ast::defFuncAST*>(op);
    if(fop->getOutputSize() > 0)
    out<<fop->getReturnType().getSID();
    else out<<"void";
    out<<" "<<fop->funcID<<"(";
    if(fop->nArgs()>0){
        out<<fop->arg().getType().getSID()<<" arg"<<fop->arg().getTraceID();
        auto & outputs = fop->getEntry().getOutputs();
        for(auto &it = outputs.begin()+1; it!=outputs.end(); it++ ){
            out<<", "<<(*it).getType().getSID()<<" arg"<<(*it).getTraceID();
        }
    }
    out<<") ";
}

void cppTranslationRule::writeBinaryAST(cgstream &out, lgf::operation *op){
    auto bop = dynamic_cast<lgf::ast::binaryExprAST*>(op);
    out.printIndent();
    out<<declVar(bop->output())<<" = "<<symbolicID(bop->lhs())<<bop->bop<<symbolicID(bop->rhs());
    out<<";\n";
}

void cppTranslationRule::writeFuncCallAST(cgstream &out, lgf::operation *op){
    auto fop = dynamic_cast<lgf::ast::callFuncAST*>(op);
    out<<declVar(fop->output())<<" = "<<fop->funcID<<"()\n";
}
} // namespace codegen