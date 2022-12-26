
#include "aog.h"

using namespace aog;
void operation::addContext(context *_ctx) { 
    ctx = _ctx;
    traceID = ctx->ops_counter++;
    assignTraceIDToOutput();
}

void operation::assignTraceIDToOutput(){
    for(auto e : getOutEdges()){
        dynamic_cast<element*>(e)->assignTraceID(ctx->elem_counter++);
    }
}