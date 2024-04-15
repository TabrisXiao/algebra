
#include <unordered_set>
#include <algorithm>
#include "lgf/node.h"
using namespace lgf;

void value::print() { global::stream::getInstance()<<represent()<<"\n"; };
//---------------------------------------------------

//////////////////////////////////////////////////////

std::string node::inputs_sid(){
    if(get_input_size() == 0) return "";
    printer p;
    p<<inputs[0].get_dual_node()->get_value().get_sid();
    for(auto iter = inputs.begin()+1; iter != inputs.end(); iter++){
        p<<", "<<(*iter).get_dual_node()->get_value().get_sid();
    }
    return p.dump();
}

//---------------------------------------------------
void node::print(){
    global::stream::getInstance().printIndent();
    global::stream::getInstance() << represent()<<"\n";
}

//---------------------------------------------------
void node::assign_value_id(int& n){
    _v_.get()->set_sid("%"+std::to_string(n));
}

//---------------------------------------------------
size_t node::get_input_size() {
    return inputs.size();
}

//////////////////////////////////////////////////////

void graph::replace_node(node* old, node* new_op){
    auto iter = std::find(nodes.begin(), nodes.end(), old);
    if(iter == nodes.end()) return;
    *iter = new_op;
}

void graph::print() {
    global::stream::getInstance().printIndent();
    std::string code = represent();
    // add space if the represent is not empty
    // {} no reprsent, shoudn't have space
    // module {}, have represent "module", should have space
    // between "module" and the {}.
    if(!code.empty()) code += " "; 
    global::stream::getInstance()<<code;
    int id = 0;
    print_graph(id);
}
//---------------------------------------------------

void graph::print_graph(int& id_start) {
    global::stream::getInstance()<<"{\n";
    global::stream::getInstance().incrIndentLevel();

    walk([this, &id_start](node* op){
        op->assign_value_id(id_start);
        id_start++;
        op->print();
    }, 1);
    global::stream::getInstance().decrIndentLevel();
    global::stream::getInstance().printIndent();
    global::stream::getInstance()<<"}\n";
}

void graph::assign_id(int n0 ){
    int n = n0;
    walk([this, &n](node* op){
        op->assign_value_id(n);
        n++;
    }, 1);
}
//---------------------------------------------------

bool graph::clean()
{
    bool check = 0;
    for(auto iter = nodes.begin(); iter!=nodes.end(); )
    {
        node* op =(*iter);
        if(op->is_deprecate()){
            
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