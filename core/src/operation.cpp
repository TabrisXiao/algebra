#include "operation.h"
#include <unordered_set>

using namespace aog;

value::value(operation * op, int id) : defOp(op), iid(id) {}
value::value(const value & val) : objInfo(val), iid(val.getIID()){
    defOp = val.getDefiningOp();
}

std::string value::represent() const {
    printer p;
    p<<"%";
    std::string id = getTraceID() > -1 ? std::to_string(getTraceID()) :"#";
    p<<id;
    p<<" <"<<getTypeID()<<">";
    return p.dump();
}

void value::print() const { global::stream::getInstance()<<represent()<<"\n"; };

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

void value::dropUser(operation* op){
    auto d = defOp->atDependency(iid);
    d->dropConnectionTo(op);
}

// void operation::setTraceIDToOutput(context *ctx){
//     for(auto i =0; i<values.size(); i++){
//         values[i].setTraceID(ctx->elem_counter++);
//     }
// }

void operation::print(){
    printOp();
}

void operation::printOp() {
    global::stream::getInstance().printIndent();
    global::stream::getInstance() << represent();
    global::stream::getInstance() <<"\n";
}

dependency * operation::atDependency(int iid) {
    // Note: Here we have to search through outputs instead of outEdges
    // as the edge might not have been connected yet. 
    for(auto i=0; i<outputs.size(); i++){
        if(outputs[i].checkIID(iid)) return &outputs[i];
    }
    return nullptr;
}

value * operation::createValue(){
    outputs.push_back(dependency(this, getOutputSize()));
    return outputs.back().atValue();
}

const value& operation::getInput(int n) const {
    auto d = dynamic_cast<dependency*>(inEdges[n]);
    return d->getValue();
}

void operation::assignValueID(int& n){
    for(auto& d : outputs){
        auto val = d.atValue();
        val->setTraceID(n++);
    }
}

void operation::dropInput(value &val){
    // assuming there's only only
    auto d = val.atDependency();
    d->dropConnectionTo(this);
}

void operation::detach(){
    // detach all inputs
    for(auto & e : inEdges){
        auto d = dynamic_cast<dependency*>(e);
        d->dropConnectionTo(this);
    }
    // detach users from this value contained in this operation;
    for(auto d : outputs){
        d.dropAllConnectionTo();
    }
}

void region::printRegion() {
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
