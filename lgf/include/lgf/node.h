
#ifndef node_H_
#define node_H_
#include "object.h"
#include "value.h"
#include <unordered_set>
#include <queue>
#include <map>
#include <set>
#include <string>
#include "printer.h"
#include "global.h"
#include "exception.h"
#include <memory>
#include <algorithm>
#include "utils.h"

// logic graph frameworks
namespace lgf{
class context;
class dependence;
class valueRef;
class graph;
class normalizer;

typedef size_t id_t;

template<typename obj>
obj& get_vec_elem_with_check(size_t n, std::vector<obj>& vec){
    auto vector_size = vec.size();
    CHECK_CONDITION(n<vector_size, "The query index exceeded the vector size.");
    return vec[n];
}

class node : public graphObject{
public : 
    // the first output value is dependency value used to store 
    // the dependency inform that don't have value connections
    node(std::string id="op", graph * g=nullptr) : 
    graphObject(id){ 
        _v_ = std::make_unique<value>(this);
        graph_ = g;
    };
    virtual ~node() = default;
    virtual std::string represent() {
        printer p;
        p<<_v_->get_sid()<<" = "<<get_sid() <<" : "<<inputs_sid();
        return p.dump();
    }
    virtual std::string inputs_sid();
    virtual void print();

    void dlink_value(value* v){
        auto iter = std::find(inputs.begin(), inputs.end(), v);
        if(iter!=inputs.end()) inputs.erase(iter);
    }

    void link_value(value* v){
        inputs.push_back(v);
    }

    void swap_input_value(value* from, value* to){
        auto iter = std::find(inputs.begin(), inputs.end(), from);
        if(iter!=inputs.end()) *iter = to;
    }
    
    template <typename... ARGS>
    void register_input(ARGS ...args)
    {
        // inputs are suppose to be value type.
        auto values = {args...};
        
        for (auto val : values)
        {
            if(val->get_defining_op() == this) {
                WARNING("Skipped register the input causing cycle dependence!");
                continue;
            }
            val->link_node(this);
            link_value(val);
        }
    }

    template <typename... ARGS>
    void accept(ARGS ...args)
    {
        auto ops = {args...};
        for (auto op : ops)
        {
            register_input(op->output());
        }
    }

    void register_input(std::vector<value*> &args){
        for(auto ptr : args){
            register_input(ptr);
        }
    }

    void replace_input(std::vector<value*>::iterator iter, value* new_n){
        if(iter == inputs.end()) return;
        (*iter)->dlink_node(this);
        new_n->link_node(this);
        *iter = new_n;
    }

    void replace_input(value* old, value* new_val){
        auto iter = std::find(inputs.begin(), inputs.end(), old);
        replace_input(iter, new_val);
    }

    void replace_input(size_t j, value* new_val){
        if(j> inputs.size()-1) return; 
        auto iter = inputs.begin()+j;
        replace_input(iter, new_val);
    }

    void replace_input(std::vector<value*>::iterator iter, node* new_n){
        replace_input(iter, new_n->output());
    }

    void replace_input(node* old, node* new_val){
        replace_input(old->output(), new_val->output());
    }

    void replace_input(size_t j, node* new_val){
        replace_input(j, new_val->output());
    }

    void replace_by(node* new_op);

    // register the input at the given position. Other inputs after 
    // that index will be push back by 1 pos.
    void register_input_at( value* val, size_t pos);

    // drop all inputs to this node, and remove all connects
    // associated to the op.
    void drop_all_inputs();

    void erase(){
        drop_all_inputs();
        // drop users of output values from this op
        _v_.get()->deprecate();
        set_removable();
    }

    std::vector<value*>& get_inputs(){ return inputs;}

    value* output(){return _v_.get();}
    value* input(size_t n=0) {return inputs[n];}
    size_t get_input_size() const;
    void set_nontrivial(){ bTrivial = 0; }
    bool is_trivial(){ return bTrivial; }

    void assign_value_id(int& n);

    // detach the input edges from this node and disconnect
    // the outputs from their users, this node still connecting to
    // the output edges.
    // for example, detach v3 will leads to 
    //    v1                       v1
    //   / \                      /
    //  v2  v3    detach v3:     v2      v3
    //     /                              |
    //    v4                        v4
    //node* detach();

    void set_activation(bool a ){ bActive = a; }
    bool is_active(){return bActive; }

    void set_exploration(bool a){ bExplored = a; }
    bool is_explored(){ return bExplored; }

    bool is_removable(){ return bRemove; }
    void set_removable(){ 
        bRemove = 1; }
    //void erase(){ detach(); bRemove = 1;}

    graph* get_parent_graph(){return graph_;}
    void set_parent_graph(graph* g){ graph_ = g; }

    std::string get_op_represent(){
        // get representation of this node after the first "="
        auto code = represent();
        auto pos = code.find("=");
        if(pos!= std::string::npos) return code.substr(pos+1);
        else return code;
    }

    bool is_identical(node* target){
        if(this == target) return true;
        if(target == nullptr) return false;
        auto code1 = this->get_op_represent();
        auto code2 = target->get_op_represent();
        if(code1!=code2) return false;
        return true;
    }

    private:
    std::vector<value*> inputs;
    std::unique_ptr<value> _v_;
    // this function is used to determine if this node contained
    // a region. If an op contained a region, it should override
    // this function.
    //virtual graph* getSubgraph(){ return nullptr;}
    bool bActive = 1;
    bool bExplored = 0;
    bool bTrivial = 1;

    // this is a member used to remove the node efficiently. 
    // Should be used solely for removing process in graph.
    bool bRemove = 0;
    graph* graph_ = nullptr;
};

class graph : public node{
    public : 
    graph() = default;
    graph( std::string id, graph* pg = nullptr )
    : node(id, pg) {}
    virtual void print() override;
    virtual std::string represent(){ return "";} 
    // return how many nodes graph contained
    size_t get_node_size(){ return nodes.size(); }
    // A breadth-first walk function that is graph modification safe.
    // Call the callable at the begining of visiting each vertex.
    // The callable should return void.
    // The fn will be executed on each node at most once. 
    // The node ran by fn is marked as done. A node will
    // got processed only if all inputs nodes are done (All
    // dependences are processed).
    // notice that this walk skipped the entryOp so that we don't 
    // need to worry about the entry op got modified by accident.
    template<typename callable>
    void walk(callable && fn, bool deepWalk=0){
        std::queue<node *> _vq;
        std::vector<node *> vertice_buffer;
        vertice_buffer.reserve(get_node_size());
        for(auto node : nodes){
            if(node->get_input_size() != 0) continue;
            _vq.push(node);
        }
        while (_vq.size())
        {
            auto v = _vq.front();
            _vq.pop();
            if(!(v->is_active())){
                continue;
            }
            if(v->is_removable() || v->is_explored()) continue;
            v->set_exploration(true);
            
            vertice_buffer.push_back(v);
            auto val = v->output();
            for(auto vn : val->get_users()){
                if (vn->is_explored() || vn->is_removable()) continue;
                _vq.push(vn);
            }
            
            if(deepWalk){
                if(auto g = dynamic_cast<graph*>(v)){
                    g->walk(fn, 1);
                }
            }
            fn(v);
        }
        for (auto v : vertice_buffer)
        {
            v->set_exploration(false);
        }
        clean();
        return;
    }

    graph* get_graph() {return dynamic_cast<graph*>(this);}
    virtual void print_graph();

    void assign_id(int n=0);

    // clean will remove all nodes marked as removable;
    // return 0 if no ops got removed. Otherwise return 1;
    bool clean();

    // this function sort the nodes in a order that the op depends on
    // others will always behind its inputs.
    //void sortByDepdency();

    
    std::vector<node*> & get_nodes(){return nodes;}
    private:
    std::vector<node*> nodes;
    // how many nodes contained in this graph
};

//----------------------------------------

}

#endif