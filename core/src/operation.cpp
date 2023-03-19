#include "operation.h"
#include "utility.h"

using namespace aog;

void objInfo::setTraceID(context * ctx){ setTraceID(ctx->ops_counter++);}

void objInfo::printIndent(context *ctx){
    utility::indent(ctx->curIndent, Xos);
}

operation* element::getDefiningOp(){return defOp;}

std::vector<operation*> element::getUsers() {
    std::vector<operation*> users;
    auto & ops = defOp->getOutVertices();
    for(auto op : ops){
        auto & elems = dynamic_cast<operation*>(op)->getInputs();
        for(auto e : elems){
            if(e == this){
                users.push_back(dynamic_cast<operation*>(op));
                break;
            }
        }
    }
    return users;
}

element::element(operation * op) : defOp(op){}

void operation::setTraceIDToOutput(context *ctx){
    for(auto i =0; i<elements.size(); i++){
        elements[i].setTraceID(ctx->elem_counter++);
    }
}

void operation::print(context *ctx){
    ctx->resetCounts();
    printOp(ctx);
}

void region::printRegion(context *ctx){
    Xos<<"{\n";
    utility::Indent idlv(ctx->curIndent);
    printOps(ctx);
    Xos<<"}\n";
}