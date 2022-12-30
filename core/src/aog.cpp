
#include "aog.h"
#include "dgraph.h"
#include "utility.h"


using namespace aog;

void objInfo::setTraceID(){setTraceID(ctx->ops_counter++);}

void objInfo::printIndent(){
    utility::indent(ctx->curIndent, Xos);
}

operation* element::getDefiningOp(){return dynamic_cast<operation*>(getSender());}

void operation::setContext(context *_ctx) { ctx = _ctx; }

template<class opType>
opType* element::getDefiningOp(){return dynamic_cast<opType*>(getSender());}

void operation::setTraceIDToOutput(){
    for(auto e : getOutEdges()){
        dynamic_cast<element*>(e)->setTraceID(ctx->elem_counter++);
    }
}

void context::print(){
    Xos<<"Context {"<<std::endl;
    utility::Indent idlv(curIndent);
    for(auto op : _ops){
        utility::indent(curIndent, Xos);
        //Xos<<"%"<<ops_counter++<<" : ";
        op->print();
    }
    Xos<<"}\n";
}

void region::printRegion(){
    Xos<<"{";
    utility::Indent idlv(ctx->curIndent);
    Xos<<"}\n";
}