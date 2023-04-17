#include "operation.h"
#include <unordered_set>

using namespace aog;

value::value(operation * op, int id) : defOp(op), iid(id) {}

std::vector<operation*> value::getUsers() const {
    std::unordered_set<operation*> buffer;
    if(auto dep = atDependency()){
        for(auto &vtx : dep->getVerticesTo()){
            buffer.insert(dynamic_cast<operation*>(vtx));
        }
    }
    std::vector<operation*> users(buffer.begin(), buffer.end());
    return users;
}

dependency * value::atDependency() const
{
    if(!defOp) return nullptr;
    return defOp->atDependency(iid);
}

// void operation::setTraceIDToOutput(context *ctx){
//     for(auto i =0; i<values.size(); i++){
//         values[i].setTraceID(ctx->elem_counter++);
//     }
// }

void operation::print(){
    printOp();
}

void region::printRegion(){
    global::stream::getInstance()<<"{\n";
    global::stream::getInstance().incrIndentLevel();
    printOps();
    global::stream::getInstance()<<"}\n";
    global::stream::getInstance().decrIndentLevel();
}

void region::assignID(int n0 ){
    int n = n0;
    walk([&](operation* op){
        op->assignValueID(n);
    });
}

inline void region::printOps(){
    BFWalk([&](dgl::vertex* _op){
        if(auto op = dynamic_cast<operation*>(_op)){
            op->printOp();
        }
    });
}

moduleOp::moduleOp(){
    setTypeID("module");
}
