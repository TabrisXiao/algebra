
#ifndef CODEDGEN_LGSKETCH_H
#define CODEDGEN_LGSKETCH_H

#include "lgf/operation.h"
#include "codegen/codeWriter.h"
namespace lgf{

namespace codegen{
enum sketchASTKind{
    kind_module,
    kind_opDef,
    kind_opDefBuilder,
    kind_typeDef,
    kind_code,
    kind_unknown
};
class sketchASTBase {
    public : 
    sketchASTBase (sketchASTKind k): kind(k) {}
    virtual ~sketchASTBase() {}
    sketchASTKind getKind(){ return kind; }
    sketchASTKind kind;
};

class sketchCodeAST : public lgf::operation, public sketchASTBase{
    public:
    sketchCodeAST(std::string & s): sketchASTBase(kind_code){
        spell = s;
    }
    std::string spell;
};

class sketchModuleAST : public lgf::graph, public sketchASTBase {
    public:
    sketchModuleAST(std::string name_="")
    : sketchASTBase(kind_module)
    , name(name_)
    , graph("sketch::moduleAST") {}
    void setName(std::string &name_){name = name_;}

    virtual std::string represent(){
        printer p;
        p<<getSID()<<" : "<<name;
        return p.dump();
    }

    void addOperationHeader(){
        if(inclOp) return;
        inclOp = 1;
        includes.push_back("lgf/operation.h");
    }
    void addIncludes(std::string inc){
        for(auto & file: includes)
            if(file== inc) return;
        includes.push_back(inc);
    }
    std::string name;
    bool inclOp = 0;
    std::vector<std::string> includes;
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

class typeDefAST : public lgf::operation, public sketchASTBase {
    public:
    typeDefAST(std::string name) 
    : sketchASTBase(kind_typeDef)
    , typeSID(name) { }

    void addParameter(std::string sid, std::string type){
        para.push_back(std::pair<std::string, std::string>(sid, type));
    }
    std::vector<std::pair<std::string, std::string>> & getParameters(){
        return para;
    }
    virtual std::string represent(){
        // todo
        return "";
    }
    std::string typeSID;
    // the parameters stored in the type variable
    // each parameter is stored as pair: name, type
    // example: x , int
    //          y , std::vector<int>
    std::vector<std::pair<std::string, std::string>> para;
};

} // namespace codegen
} // namespace lgf

#endif