
#include <unordered_set>
#include <algorithm>
#include "lgf/operation.h"
using namespace lgf;

value::value(operation * op) : defop(op) {
}
//---------------------------------------------------

value::value(operation * op, type_t type, std::string sid_) 
: defop(op), vtp(type) {
    setSID(sid_);
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
    return users;
}
//---------------------------------------------------

void value::dropUser(operation *op){
    auto &iter = std::find(users.begin(), users.end(), op);
    if(iter==users.end()) return;
    auto user = (*iter);
    auto & userInputs = user->getInputs();
    auto & del = std::find(userInputs.begin(), userInputs.end(),this);
    if(del!=userInputs.end()) userInputs.erase(del);
    users.erase(iter);
}
//---------------------------------------------------

void value::dropUsers(){ 
    for(auto op : users){
        dropUser(op);
    }
}

//---------------------------------------------------
std::unique_ptr<value>* value::getPtr(){
    auto op = getDefiningOp();
    for(auto & output : op->getOutputs()){
        if(this == output.get()){
            return &output;
        }
    }
    return nullptr;
}

//////////////////////////////////////////////////////

std::string operation::representInputs(){
    if(getInputSize() == 0) return "";
    printer p;
    auto ins = getInputs();
    p<<ins[0]->represent();
    for(auto iter = ins.begin()+1; iter != ins.end(); iter++){
        p<<", "<<(*iter)->represent();
    }
    return p.dump();
}
//---------------------------------------------------

std::string operation::representOutputs(){
    // this function print the outputs in the form:
    // %1, %2 = 
    if(outputs.size()<2) return "";
    printer p;
    // skip the 1st dependencyValue
    p<<outputs[1]->represent();
    for(auto iter =outputs.begin()+2; iter!=outputs.end(); iter++){
        p<<", "<<(*iter)->represent();
    }
    return p.dump();
}
//---------------------------------------------------

void operation::print(){
    global::stream::getInstance().printIndent();
    //printOutputs();
    global::stream::getInstance() << represent()<<"\n";
    //if(auto g = expandToGraph()){
    //    global::stream::getInstance() <<" ";
    //    g->print();
    //} else global::stream::getInstance() <<"\n";
}
//---------------------------------------------------

void operation::registerInputAt( value* val, int pos){
    inputs.insert(inputs.begin()+pos, val);
}
//---------------------------------------------------

value* operation::createValue(){
    auto val = std::make_unique<value>(this);
    outputs.push_back(std::move(val));
    return outputs.back().get();
}
value* operation::createValue(type_t& type, std::string sid){
    auto val = std::make_unique<value>(this, type, sid);
    outputs.push_back(std::move(val));
    return outputs.back().get();
}
//---------------------------------------------------

void operation::assignValueID(int& n){
    for(auto &val : outputs){
        if(dynamic_cast<dependencyValue*>(val.get()))
            continue;
        val->setTraceID(n++);
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
    for(auto input: inputs){
        input->dropUser(this);
    }
    inputs.clear();
}
//---------------------------------------------------

bool operation::isDependencyFullfilled(){
    for(auto input : inputs){
        if(!input->getDefiningOp()->isExplored()) return false;
    }
    return true;
}
//---------------------------------------------------

graph* operation::expandToGraph()
{ 
    return dynamic_cast<graph*>(this);
}
//---------------------------------------------------

void operation::replaceInputValue(int n, value* v){
    if(n+1 > inputs.size())return;
    inputs[n]->dropUser(this);
    // update the corresponding valueRef to the new one
    inputs[n]=v;
    v->addUsesr(this);
}
//---------------------------------------------------

void operation::replaceBy(operation* new_op){
    auto output_size = getOutputSize();
    CHECK_VALUE(output_size, new_op->getOutputSize(), "New op must have the same number of outputs as the original op.");
    // assume that the inputs are settled down for the new op,
    // here we only substitue the users for outputs.
    // it also assume the output size is the same as the old one.
    for(auto i=0; i<outputs.size(); i++){
        auto output = outputs[i].get();
        auto& users = output->getUsers();
        for(auto &user : users){
            user->inputs[i] = new_op->outputs[i].get();
        }
        output->getUsers().clear();
    }
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
    for(auto & op : nodes){
        if(op->isRemovable() || !op->isActive()) continue;
        op->print();
    }
    global::stream::getInstance().decrIndentLevel();
    global::stream::getInstance().printIndent();
    global::stream::getInstance()<<"}\n";
}

void graph::assignID(int n0 ){
    int n = n0;
    for(auto & op : nodes){
        op->assignValueID(n);
        if(auto g = dynamic_cast<graph*>(op)){
            int gn = 0;
            int entryn = 0;
            g->getEntry().assignValueID(entryn);
            g->assignID(gn);
        }
    }
    // walk([&](operation* op){
    //     op->assignValueID(n);
    //     if(auto g = dynamic_cast<graph*>(op)){
    //         int gn = 0;
    //         int entryn = 0;
    //         g->getEntry().assignValueID(entryn);
    //         g->assignID(gn);
    //     }
    // }, 1);
}
//---------------------------------------------------

void graph::registerOp(operation* op){
    op->setParentGraph(this);
    if(op->getInputSize() == 0) op->appendTo(dynamic_cast<operation*>(&entry)); 
    nodes.push_back(op);
}
//---------------------------------------------------

void graph::clean()
{
    for(auto iter = nodes.begin(); iter!=nodes.end(); )
    {
        operation* op =(*iter);
        if(op->isRemovable()){
            delete op;
            iter = nodes.erase(iter);
        } else if (auto g = dynamic_cast<graph*>(op)){
            g->clean();
        }
        iter++;
    }
}
//---------------------------------------------------