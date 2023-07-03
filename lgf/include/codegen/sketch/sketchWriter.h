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
        if(typeInclude) prefix = "lgf::value& ";
        if(vec.size()==0) return "";
        res+=prefix+vec[0].getSID();
        for(auto i=1; i<vec.size(); i++){
            res+=", "+prefix+vec[i].getSID();
        }
        return res;
    }
    void printTypeGuards(cgstream &out, std::vector<value>& vec ){
        if(vec.size() == 0) return;
        out.printIndent();
        auto & elm = vec[0];
        out<<elm.getSID()<<".type_guard<"+elm.getType().getSID()<<">();\n";
        for(auto i=1; i<vec.size(); i++){
            auto & elm = vec[i];
            out.printIndent();
            out<<elm.getSID()<<".type_guard<"+elm.getType().getSID()<<">();\n";
        }
    }
    std::string printValueTypeArraySignature(std::vector<value>& types){
        std::string res;
        if(types.size()==0) return res;
        std::string prefix = types[0].getType().getSID()+" ";
        res += prefix+types[0].getSID()+"_t";
        for(auto i=1; i<types.size(); i++){
            res+=", "+prefix+types[i].getSID()+"_t";
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
        out<<builder->getOpName()<<"(){}\n";
        auto outputType = printValueTypeArraySignature(vecOutput);
        auto str = printValueArraySignature(vecInput, 1);
        // write build function:
        out.printIndent();
        out<<"static "<<builder->getOpName()<<"* build("<<outputType;
        if(!str.empty()) {
            out<<", "<<str;
        }
        out<<"){\n";
        {
            indentGuard a(out);
            if(ninput>0){
                printTypeGuards(out, vecInput);
            }
            out.printIndent();
            out<<"auto op = new "<<builder->getOpName()<<"();\n";
            if(ninput>0){
                out.printIndent();
                out<<"op->registerInput("<<printValueArraySignature(vecInput)<<");\n";
            } 
            out.printIndent();
            out<<"op->setSID(\""<<op->getSID()<<"\");\n";
            
            
            for(auto i=0; i<noutput; i++){
                out.printIndent();
                auto & val = vecOutput[i];
                out<<"op->createValue("<<val.getSID()<<"_t, \""<<val.getSID()<<"\");\n";
            }
            out.printIndent();
            out<<"return op;\n";
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
    void writeSketchCodeAST(cgstream &out, lgf::operation *op){
        auto spellOp = dynamic_cast<sketchCodeAST*>(op);
        out<<spellOp->spell<<"\n";
    }
    void writeTypeDefAST(cgstream &out, lgf::operation *op_){
        auto op = dynamic_cast<typeDefAST*>(op_);
        out<<"class "<<op->typeSID<<": public ";
        if(op->parents.size()==0){
            out<<"lgf::type_t {\n";
        }else {
            out<<op->parents[0];
            for(auto i=1; i<op->parents.size(); i++){
                out<<", public "<<op->parents[i];
            }
            out<<" {\n";
        }
        indentGuard a(out);
        out.printIndent();
        out<<"public:\n";
        out.printIndent();
        out<<op->typeSID<<"() { id=\""<<op->typeSID<<"\"; }\n";
        out.printIndent();
        out<<"static "<<op->typeSID<<" build(";
        auto & para = op->getParameters();
        auto size = para.size();
        if(size>0){
            out << para[0].second<<" "<<para[0].first;
            for(auto i = 1; i<size; i++){
                out<<", "<< para[i].second<<" "<<para[i].first;
            }
        }
        out<<"){\n";{
            indentGuard g1(out);
            out.printIndent();
            out<<op->typeSID<<" obj;\n";
            if(size>0){
                out.printIndent();
                out<<"obj."<<para[0].first<<"_="<<para[0].first<<";\n";
                for(auto i = 1; i<size; i++){
                    out.printIndent();
                    out<<"obj."<<para[i].first<<"_="<<para[i].first<<";\n";
                } 
            }
            out.printIndent();
            out<<"return obj;\n";
        }
        out.printIndent();
        out<<"}\n";
        int n= 0;
        for(auto & pair : op->getParameters()){
            out.printIndent();
            out<<"const "<<pair.second<<"& "<<pair.first<<"(){ return "<<pair.first<<"_; }\n";
        }
        for(auto & pair : op->getParameters()){
            out.printIndent();
            out<<pair.second<<" "<<pair.first<<"_;\n";
        }
        out<<"};\n\n";
    }
    virtual void writeHeader(cgstream &out, lgf::graph* reg){
        auto module = dynamic_cast<sketchModuleAST*>(reg);
        auto name = module->name;
        std::string scopeName = "";
        bool isDone = 0;
        for(auto & c : name){
            if(c=='.') {
                c='_';
                isDone = 1;
            }
            if(!isDone) scopeName += c;
            c = std::toupper(c);
        }
        out<<"#ifndef "+name+"_H\n";
        out<<"#define "+name+"_H\n";
        for(auto & path : module->includes){
            out<<"#include \""+path+"\"\n";
        }
        out<<"\n";
        out<<"namespace "+scopeName+"{\n";
    }
    virtual void writeFinal(cgstream &out, lgf::graph* reg){
        out<<"\n}\n";
        out<<"#endif\n";
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
            case kind_typeDef:
                writeTypeDefAST(out, op);
                return 1;
            case kind_code:
                writeSketchCodeAST(out, op);
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