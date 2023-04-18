#include "operation.h"
#include <unordered_set>

using namespace aog;

value::value(operation * op, id_t id) : defOp(op), iid(id) {}
//---------------------------------------------------

value::value(const value & val) : objInfo(val), iid(val.getIID()){
    defOp = val.getDefiningOp();
}
//---------------------------------------------------

std::string value::represent() const {
    printer p;
    p<<"%";
    std::string id = getTraceID() > -1 ? std::to_string(getTraceID()) :"#";
    p<<id;
    p<<" <"<<getTypeID()<<">";
    return p.dump();
}
//---------------------------------------------------

void value::print() const { global::stream::getInstance()<<represent()<<"\n"; };
//---------------------------------------------------

std::vector<operation*> value::getUsers() const {
    std::unordered_set<operation*> buffer;
    if(auto dep = getDependence()){
        for(auto &vtx : dep->getVerticesTo()){
            buffer.insert(dynamic_cast<operation*>(vtx));
        }
    }
    std::vector<operation*> users(buffer.begin(), buffer.end());
    return users;
}
//---------------------------------------------------

dependence * value::getDependence() const
{
    if(!defOp) return nullptr;
    return defOp->getDependence(iid);
}
//---------------------------------------------------

void value::dropUser(operation* op){
    auto d = defOp->getDependence(iid);
    d->dropConnectionTo(op);
}
//---------------------------------------------------

void value::replaceBy(const value & val){
    auto d = getDependence();
    auto dnew = val.getDependence();
    d->replaceBy(dnew);
}

//////////////////////////////////////////////////////

void dependence::replaceBy(dependence *dep){
    for(auto& vtx : verticesTo){
        auto op = dynamic_cast<operation*>(vtx);
        op->replaceInputDependence(this, dep);
    }
}

//////////////////////////////////////////////////////

// void operation::setTraceIDToOutput(context *ctx){
//     for(auto i =0; i<values.size(); i++){
//         values[i].setTraceID(ctx->elem_counter++);
//     }
// }

void operation::print(){
    printOp();
}
//---------------------------------------------------

void operation::printOp() {
    global::stream::getInstance().printIndent();
    global::stream::getInstance() << represent();
    global::stream::getInstance() <<"\n";
}
//---------------------------------------------------

dependence * operation::getDependence(id_t iid) {
    // Note: Here we have to outputs instead of outEdges
    // as the edge might not have been connected yet. 
    // The outputs must be ordered based on the iid
    // from small to large.
    if(iid >= outputs.size()) return nullptr;
    return &outputs[iid];
}
//---------------------------------------------------

value * operation::createValue(){
    outputs.push_back(dependence(this, getOutputSize()));
    return outputs.back().getValue();
}
//---------------------------------------------------

const value& operation::inputRef(int n) const {
    auto d = dynamic_cast<dependence*>(inEdges[n]);
    return *(d->getValue());
}
//---------------------------------------------------

void operation::assignValueID(int& n){
    for(auto& d : outputs){
        auto val = d.getValue();
        val->setTraceID(n++);
    }
}
//---------------------------------------------------

id_t operation::getInputSize() const {
    return inEdges.size();
}
//---------------------------------------------------

id_t operation::getOutputSize() const {
    return outEdges.size();
}
//---------------------------------------------------

void operation::dropInput(value &val){
    // assuming there's only only
    auto d = val.getDependence();
    d->dropConnectionTo(this);
}
//---------------------------------------------------

operation* operation::detach(){
    // detach all inputs
    for(auto & e : inEdges){
        auto d = dynamic_cast<dependence*>(e);
        d->dropConnectionTo(this);
    }
    // detach users from this value contained in this operation;
    for(auto d : outputs){
        d.dropAllConnectionTo();
    }
    return this;
}
//---------------------------------------------------

void operation::replaceInputDependence(dependence* d1, dependence* d2 ){
    auto iter = std::find(inEdges.begin(), inEdges.end(), d1);
    if(iter == inEdges.end()) return;
    (*iter)->eraseVertexTo(this);
    *iter = d2;
}
//---------------------------------------------------

void operation::replaceInputValue(value & oldV, value & newV){
    auto d1 = oldV.getDependence();
    auto d2 = newV.getDependence();
    if(!d1 || !d2) return;
    replaceInputDependence(d1, d2);
}

//////////////////////////////////////////////////////

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
