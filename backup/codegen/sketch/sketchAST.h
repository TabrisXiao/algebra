
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
    kind_funcDef,
    kind_funcCall,
    kind_represent,
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
    sketchCodeAST(): sketchASTBase(kind_code){}
    static sketchCodeAST* build(std::string code){
        auto op = new sketchCodeAST();
        op->spell = code;
        return op;
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

    static sketchModuleAST* build(std::string name_=""){
        auto op = new sketchModuleAST(name_);
        return op;
    }

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
    static opDefBuilderAST* build(){
        auto op = new opDefBuilderAST();
        return op;
    }
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
    std::vector<std::string> parents;
};

class opDefAST : public lgf::graph, public sketchASTBase {
    public:
    opDefAST(std::string name)
    : sketchASTBase(kind_opDef)
    , graph("sketch::DefOpAST") {
        ioOp.setOpName( name );
        addOp(&ioOp);
    }
    static opDefAST* build(std::string name){
        auto op = new opDefAST(name);
        return op;
    }
    opDefBuilderAST* getBuilderOp(){
        return &ioOp;
    }

    void addSubgraph(graph* g){
        addOp(dynamic_cast<operation*>(g));
    }
    void addParent(std::string p ){
        ioOp.parents.push_back(p);
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

    static typeDefAST* build(std::string name){
        auto op = new typeDefAST(name);
        return op;
    }

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
    void addParent(std::string pr){
        parents.push_back(pr);
    }
    std::string typeSID;
    // the parameters stored in the type variable
    // each parameter is stored as pair: name, type
    // example: x , int
    //          y , std::vector<int>
    std::vector<std::pair<std::string, std::string>> para;
    std::vector<std::string> parents;
};

class representAST : public lgf::operation, public sketchASTBase {
    public:
    representAST() : sketchASTBase(kind_represent) {}
    representAST & operator <<(std::string & cc){
        content+=cc;
        return *this;
    }
    std::string content;
};

struct variable{
    std::string id, type, value;
};

class funcDefAST: public lgf::graph, public sketchASTBase {
    public:
    funcDefAST(std::string id, std::string rtype, std::string type = "") 
    : sketchASTBase(kind_funcDef)
    , graph("sketch::funcDefAST")
    , funcId(id)
    , funcType(type)
    , returnType(rtype)
    {}
    void addArg(std::string type, std::string id, std::string dvalue=""){
        args.push_back(variable({type, id, dvalue}));
    }
    std::string funcId, funcType, returnType;
    // funcType can used to specify if func is virtual, const etc.
    std::vector<variable> args;
};

class funcCallAST : public lgf::operation, public sketchASTBase {
    public:
    funcCallAST(std::string id)
    : sketchASTBase(kind_funcCall)
    , funcId(id) 
    {}
    std::string funcId;
};

} // namespace codegen
} // namespace lgf

#endif