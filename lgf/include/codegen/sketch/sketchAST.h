
#ifndef CODEDGEN_LGSKETCH_H
#define CODEDGEN_LGSKETCH_H

#include "lgf/operation.h"
#include "codegen/codeWriter.h"
namespace lgf{

namespace codegen{
enum sketchASTKind{
    kind_opDef,
    kind_opDefBuilder,
    kind_macroSpell,
    kind_unknown
};
class sketchASTBase {
    public : 
    sketchASTBase (sketchASTKind k): kind(k) {}
    virtual ~sketchASTBase() {}
    sketchASTKind getKind(){ return kind; }
    sketchASTKind kind;
};

class macroSpellAST : public lgf::operation, public sketchASTBase{
    public:
    macroSpellAST(std::string & s): sketchASTBase(kind_macroSpell){
        spell = s;
    }
    std::string spell;
};

// class cppCodeReg : public lgf::graph, public sketchASTBase{
//     public:
//     cppCodeReg() : graph("codegen::sketch::cppCodeReg") {}
// };
class opDefBuilderAST : public lgf::operation, public sketchASTBase {
    public:
    opDefBuilderAST(): sketchASTBase(kind_opDefBuilder) {}
    void addInput(std::string name, lgf::type_t tp){
        input_.push_back(value());
        input_.back().setSID(name);
        input_.back().setType(tp);
    }
    void addOutput(std::string name, lgf::type_t tp){
        output_.push_back(value());
        output_.back().setSID(name);
        output_.back().setType(tp);
    }
    void setOpName(std::string name ){ opname = name; }
    std::string getOpName(){return opname;}
    std::vector<value>& getInputSig(){return input_;}
    std::vector<value>& getOutputSig(){return output_;}
    virtual std::string represent(){ return "";}
    std::vector<value> input_;
    std::vector<value> output_;
    std::string opname;
};

class opDefAST : public lgf::graph, public sketchASTBase {
    public:
    opDefAST(std::string name)
    : sketchASTBase(kind_opDef)
    , graph("sketch::DefOpAST") {
        ioOp.setOpName( name );
        addOp(&ioOp);
    }
    opDefBuilderAST* getBuilderOp(){
        return &ioOp;
    }

    void addSubgraph(graph* g){
        addOp(dynamic_cast<operation*>(g));
    }

    virtual std::string represent(){
        printer p;
        p<<getSID()<<" : "<<ioOp.representOutputs()<<" = "<<opname;
        p<<"("<<ioOp.representOutputs()<<")";
        return p.dump();
    }
    std::string opname;
    opDefBuilderAST ioOp;
};

} // namespace codegen
} // namespace lgf

#endif