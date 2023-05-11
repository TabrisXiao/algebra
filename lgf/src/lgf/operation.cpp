
#include <unordered_set>
#include <algorithm>
#include "lgf/operation.h"
using namespace lgf;

value::value(operation * op, id_t id) : defop(op), iid(id) {
}
//---------------------------------------------------

std::string value::represent() {
    printer p;
    p<<"%"<<getSID();
    std::string id = getTraceID() > -1 ? std::to_string(getTraceID()):"#";
    p<<id;
    p<<" "<<getTR();
    return p.dump();
}
//---------------------------------------------------

void value::print() { global::stream::getInstance()<<represent()<<"\n"; };
//---------------------------------------------------

std::vector<operation*> value::getUsers() {
    std::vector<operation*> users;
    auto& outgoings = defop->getOutgoings();
    for(auto iter = outgoings.begin(); iter!=outgoings.end(); iter++){
        if(auto ref = (*iter)->inputRefByValue(*this)){
            users.push_back(*iter);
        }
    }
    return users;
}
//---------------------------------------------------

value & valueRef::getValue(){
    return defop->getValueByID(iid);
}

//////////////////////////////////////////////////////

std::string operation::representOutputs(){
    // this function print the outputs in the form:
    // %1, %2 = 
    printer p;
    p<<outputs.front().represent();
    for(auto iter =outputs.begin()+1; iter!=outputs.end(); iter++){
        p<<", "<<(*iter).represent();
    }
    return p.dump();
}
//---------------------------------------------------

void operation::print(){
    global::stream::getInstance().printIndent();
    //printOutputs();
    global::stream::getInstance() << represent();
    if(auto g = expandToGraph()){
        global::stream::getInstance() <<" ";
        g->print();
    } else global::stream::getInstance() <<"\n";
}
//---------------------------------------------------

void operation::registerInputAt( value& val, int pos){
    inputs.insert(inputs.begin()+pos, valueRef(val));
}
//---------------------------------------------------

value& operation::createValue(){
    outputs.push_back(value(this, getOutputSize()));
    return outputs.back();
}
//---------------------------------------------------

void operation::assignValueID(int& n){
    for(auto &val : outputs){
        val.setTraceID(n++);
    }
}
//---------------------------------------------------

size_t operation::getInputSize() const {
    return inputs.size();
}
//---------------------------------------------------

size_t operation::getOutputSize() const {
    return outputs.size();
}
//---------------------------------------------------

void operation::dropAllInputs(){
    for(auto op : incomings){
        op->outgoings.erase(this);
    }
    incomings.clear();
    inputs.clear();
}
//---------------------------------------------------

bool operation::isDependencyFullfilled(){
    for(auto &op : incomings){
        if(!op->isExplored()) return false;
    }
    return true;
}
//---------------------------------------------------

graph* operation::expandToGraph()
{ 
    return dynamic_cast<graph*>(this);
}
//---------------------------------------------------

void operation::replaceInputValue(int n, value& v){
    if(n+1 > inputs.size())return;
    auto op = inputs[n].getDefiningOp();
    breakLinkFrom(op);
    // update the corresponding valueRef to the new one
    inputs[n].referTo(v);
}
//---------------------------------------------------

valueRef* operation::inputRefByValue(const value & v){
    auto op = v.getDefiningOp();
    auto id  = v.getIID();
    auto iter = std::find_if(inputs.begin(), inputs.end(), 
    [&](const valueRef &ref){
        return (op==ref.getDefiningOp() && id == ref.getIID());
    });
    if(iter!=inputs.end()) return &(*iter);
    return nullptr;
}
//---------------------------------------------------

void operation::replaceBy(operation* new_op){
    auto output_size = getOutputSize();
    CHECK_VALUE(output_size, new_op->getOutputSize(), "New op must have the same number of outputs as the original op.");
    // assume that the inputs are settled down for the new op,
    // here we only substitue the users for outputs.
    for(auto iter= outgoings.begin(); iter!=outgoings.end();iter++){
        // replace the valueRef in the users;
        for(auto &val : outputs){
            if(auto ref = (*iter)->inputRefByValue(val)){
                // the new value suppose to have the same iid
                // as the old one
                ref->referTo(new_op->getValueByID(val.getIID()));
                // connecting the new op to users
                new_op->linkTo(*iter);
            }
        }
        // break connections with users
        (*iter)->incomings.erase(this);
    }
    outgoings.clear();
}

//////////////////////////////////////////////////////

void graph::print() {
    global::stream::getInstance().printIndent();
    std::string code = represent();
    // add space if the represent is not empty
    // {} no reprsent, shoudn't have space
    // module {}, have represent "module", should have space
    // between "module" and the {}.
    if(!code.empty()) code += " "; 
    global::stream::getInstance()<<code;
    printGraph();
}
//---------------------------------------------------

void graph::printGraph() {
    global::stream::getInstance()<<"{\n";
    global::stream::getInstance().incrIndentLevel();
    walk([&](operation* op){ 
        op->print();});
    global::stream::getInstance().decrIndentLevel();
    global::stream::getInstance().printIndent();
    global::stream::getInstance()<<"}\n";
}

void graph::assignID(int n0 ){
    int n = n0;
    walk([&](operation* op){
        op->assignValueID(n);
        if(auto g = dynamic_cast<graph*>(op)){
            int gn = 0;
            int entryn = 0;
            g->getEntry().assignValueID(entryn);
            g->assignID(gn);
        }
    });
}
//---------------------------------------------------

void graph::addOp(operation* op){
    nodes.push_back(op);
    op->setParentGraph(this);
    if(op->getInputSize() == 0) attachToEntrance(op); 
}
//---------------------------------------------------

void graph::clean()
{
    auto & entrances= getOutgoings();
    for(auto iter = entrances.begin(); iter!=entrances.end();){
        if((*iter)->isRemovable()) {
            iter = entrances.erase(iter);
        }
        else iter++;
    }
    for(auto iter = nodes.begin(); iter!=nodes.end(); )
    {
        operation* op =(*iter);
        if(op->isRemovable()){
            delete op;
            iter = nodes.erase(iter);
        }
        else iter++;
    }
}
//---------------------------------------------------