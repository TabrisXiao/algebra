
#ifndef CODEDGEN_LGSKETCH_H
#define CODEDGEN_LGSKETCH_H

#include "lgf/operation.h"
#include "codeWriter.h"
namespace lgf{

namespace codegen{

class sketchOp {
    public : 
    sketchOp () = default;
};

class cppCodeReg : public lgf::graph, public sketchOp{
    public:
    cppCodeReg() : graph("codegen::sketch::cppCodeReg") {}
};

class opDefAST : public lgf::graph, public sketchOp {
    public:
    opDefAST(std::string name): graph("sketch::DefOpAST") {opname = name;}
    void addInput(std::string name, lgf::type_t tp){
        auto& val = getEntry().createValue();
        val.setSID(name);
        val.setType(tp);
    }
    void addOutput(std::string name, lgf::type_t tp){
        auto& val = createValue();
        val.setSID(name);
        val.setType(tp);
    }

    void addSubgraph(graph* g){
        addOp(dynamic_cast<operation*>(g));
    }

    virtual std::string represent(){
        printer p;
        p<<getSID()<<" : "<<representOutputs()<<" = "<<opname;
        p<<"("<<getEntry().representOutputs()<<")";
        return p.dump();
    }
    std::string opname;
};

} // namespace codegen
} // namespace lgf

#endif