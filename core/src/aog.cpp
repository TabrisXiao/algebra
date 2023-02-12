
#include "aog.h"
#include "dgraph.h"
#include "utility.h"


using namespace aog;

void objInfo::setTraceID(context * ctx){ setTraceID(ctx->ops_counter++);}

void objInfo::printIndent(context *ctx){
    utility::indent(ctx->curIndent, Xos);
}

operation* element::getDefiningOp(){return defOp;}

element::element(operation * op) : defOp(op){}

template<class opType>
opType* element::getDefiningOp(){return dynamic_cast<opType*>(defOp);}

void operation::setTraceIDToOutput(context *ctx){
    for(auto i =0; i<elements.size(); i++){
        elements[i].setTraceID(ctx->elem_counter++);
    }
}

void region::printRegion(context *ctx){
    Xos<<"{\n";
    utility::Indent idlv(ctx->curIndent);
    printOps(ctx);
    Xos<<"}\n";
}