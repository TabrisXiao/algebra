#ifndef CODEGEN_SKETCHWRITER_H
#define CODEGEN_SKETCHWRITER_H

#include "codegen/codeWriter.h"
#include "sketchAST.h"

namespace lgf{
namespace codegen{
class sketch2cppTranslationRule : public translateRule<sketchASTBase>{
    public:
    sketch2cppTranslationRule(): translateRule<sketchASTBase>(){}
    void writeOpDefAST(cgstream &out, lgf::operation *op){
        auto defop = dynamic_cast<opDefAST*>(op);
        out.printIndent();
        auto opname = defop->getBuilderOp()->getOpName();
        out<<"\n// ---------- "<<opname<<" ----------\n";
        out<<"class "<<opname<<" : public lgf::operation\n";
    }
    std::string printValueArraySignature(std::vector<value>& vec, bool typeInclude = 0){
        std::string res;
        std::string prefix = "";
        if(typeInclude) prefix = "value& ";
        if(vec.size()==0) return "";
        res+=prefix+vec[0].getSID();
        for(auto i=1; i<vec.size(); i++){
            res+=", "+prefix+vec[i].getSID();
        }
        return res;
    }
    void writeOpDefBuilerdAST(cgstream &out, lgf::operation *op){
        auto builder = dynamic_cast<opDefBuilderAST*>(op);
        auto vecInput = builder->getInputSig();
        auto ninput = vecInput.size();
        auto vecOutput = builder->getOutputSig();
        auto noutput = vecOutput.size();
        // write constructor
        // opname ( arg0, arg1, ....) 
        out.printIndent();
        out<<"public:\n";
        out.printIndent();
        out<<builder->getOpName()<<"(";
        out<<printValueArraySignature(vecInput, 1);
        out<<"){\n";
        {
            indentGuard a(out);
            if(ninput>0){
                out.printIndent();
                out<<"registerInput("<<printValueArraySignature(vecInput)<<");\n";
            }
            

            for(auto i=0; i<noutput; i++){
                out.printIndent();
                auto & val = vecOutput[i];
                out<<"createValue("<<val.getSID()<<", "<<val.getType().getSID()<<");\n";
            }
        }
        out.printIndent();
        out<<"}\n";

        for(auto i=0; i<ninput; i++){
            out.printIndent();
            out<<"lgf::value& "<<vecInput.at(i).getSID()<<"(){ return inputValue("<<std::to_string(i)<<"); }\n";
        }
        
        for(auto i=0; i<noutput; i++){
            out.printIndent();
            out<<"lgf::value& "<<vecOutput.at(i).getSID()<<"(){ return outputValue("<<std::to_string(i)<<"); }\n";
        }
    }
    void writeMACROSpellAST(cgstream &out, lgf::operation *op){
        auto spellOp = dynamic_cast<macroSpellAST*>(op);
        out<<spellOp->spell<<"\n";
    }
    virtual bool write(cgstream &out, sketchASTBase *op_){
        auto op = dynamic_cast<lgf::operation*>(op_);
        switch(op_->getKind()){
            case kind_opDef:
                writeOpDefAST(out, op);
                return 1;
            case kind_opDefBuilder:
                writeOpDefBuilerdAST(out, op);
                return 1;
            case kind_macroSpell:
                writeMACROSpellAST(out, op);
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
        out<<"}";
        if(auto structOp = dynamic_cast<opDefAST*>(reg))
            out<<";";
        out<<"\n";
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
}
#endif