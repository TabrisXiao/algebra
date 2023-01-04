
#include "aog.h"
#include "dgraph.h"
#include "utility.h"


using namespace aog;

void objInfo::setTraceID(){setTraceID(ctx->ops_counter++);}

void objInfo::printIndent(){
    utility::indent(ctx->curIndent, Xos);
}

operation* element::getDefiningOp(){return defOp;}

element::element(operation * op):objInfo(op->getContext()), defOp(op){}

void operation::setContext(context *_ctx) { ctx = _ctx; }

template<class opType>
opType* element::getDefiningOp(){return dynamic_cast<opType*>(defOp);}

void operation::setTraceIDToOutput(){
    for(auto e : getOutputs()){
        e.setTraceID(ctx->elem_counter++);
    }
}

void region::printRegion(){
    Xos<<"{\n";
    utility::Indent idlv(ctx->curIndent);
    printOps();
    Xos<<"}\n";
}