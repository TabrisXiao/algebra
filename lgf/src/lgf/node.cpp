
#include <unordered_set>
#include <algorithm>
#include "lgf/node.h"
using namespace lgf;

value::value(node * op) : defop(op) {
}
//---------------------------------------------------

value::value(node * op, std::string sid_) 
: defop(op){
    set_sid(sid_);
}
//---------------------------------------------------

value::~value() {
    deprecate();
}
//---------------------------------------------------

void value::deprecate(){
    for(auto user : users){
        user->dlink_value(this);
    }
    users.clear();
}
//---------------------------------------------------

void value::remove_user(node* user){
    auto iter = std::find(users.begin(), users.end(), user);
    (*iter)->dlink_value(this);
    if(iter!=users.end()) users.erase(iter);
}
//---------------------------------------------------

std::string value::represent() {
    printer p;
    p<<"%"<<get_sid();
    p<<" "<<get_trace_id();
    //p<<" ("<<getUsers().size()<<")"; 
    //if(getUsers().size()>0) p<<" first user: "<<getUsers()[0];
    return p.dump();
}
//---------------------------------------------------

void value::print() { global::stream::getInstance()<<represent()<<"\n"; };
//---------------------------------------------------

//////////////////////////////////////////////////////

std::string node::represent_inputs(){
    if(get_input_size() == 0) return "";
    printer p;
    auto ins = get_inputs();
    p<<ins[0]->represent();
    for(auto iter = ins.begin()+1; iter != ins.end(); iter++){
        p<<", "<<(*iter)->represent();
    }
    return p.dump();
}
//---------------------------------------------------

void node::print(){
    global::stream::getInstance().printIndent();
    //printOutputs();
    global::stream::getInstance() << represent()<<"\n";
    //if(auto g = expandToGraph()){
    //    global::stream::getInstance() <<" ";
    //    g->print();
    //} else global::stream::getInstance() <<"\n";
}
//---------------------------------------------------

void node::register_input_at( value* val, size_t pos){
    inputs.insert(inputs.begin()+pos, val);
}
//---------------------------------------------------

void node::assign_value_id(int& n){
    _v_.get()->set_trace_id(n);
    n++;
}
//---------------------------------------------------

size_t node::get_input_size() const {
    return inputs.size();
}
//---------------------------------------------------

void node::drop_all_inputs(){
    for(auto input=inputs.begin(); input!=inputs.end(); input++){
        (*input)->dlink_node(this);
    }
    inputs.clear();
}
//---------------------------------------------------

void node::replace_by(node* new_op){
    if(this == new_op) return;
    _v_.get()->swap(*(new_op->output()));
}

//////////////////////////////////////////////////////

void graph::print() {
    assign_id(0);
    global::stream::getInstance().printIndent();
    std::string code = represent();
    // add space if the represent is not empty
    // {} no reprsent, shoudn't have space
    // module {}, have represent "module", should have space
    // between "module" and the {}.
    if(!code.empty()) code += " "; 
    global::stream::getInstance()<<code;
    print_graph();
}
//---------------------------------------------------

void graph::print_graph() {
    global::stream::getInstance()<<"{\n";
    global::stream::getInstance().incrIndentLevel();
    for(auto & op : nodes){
        if(op->is_removable() || !op->is_active()) continue;
        op->print();
    }
    global::stream::getInstance().decrIndentLevel();
    global::stream::getInstance().printIndent();
    global::stream::getInstance()<<"}\n";
}

void graph::assign_id(int n0 ){
    int n = n0;
    for(auto & op : nodes){
        op->assign_value_id(n);
        if(auto g = dynamic_cast<graph*>(op)){
            int gn = 0;
            int entryn = 0;
            g->assign_id(gn);
        }
    }
}
//---------------------------------------------------

bool graph::clean()
{
    bool check = 0;
    for(auto iter = nodes.begin(); iter!=nodes.end(); )
    {
        node* op =(*iter);
        if(op->is_removable()){
            
            iter = nodes.erase(iter);
            check = 1;
            delete op;
        } else if (auto g = dynamic_cast<graph*>(op)){
            check = g->clean();
            iter++;
        } else iter++;
    }
    return check;
}
//---------------------------------------------------