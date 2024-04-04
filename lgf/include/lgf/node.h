
#ifndef node_H_
#define node_H_
#include "type.h"
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
class node;
class dependence;
class value;
class valueRef;
class graph;
class normalizer;
class LGFContext;

typedef size_t id_t;


//symbolic id;
using sid_t = std::string;

class graph_object {
    public:
    graph_object() = default;
    graph_object(sid_t id) : sid(id) {}
    std::string get_sid() const { return sid; }
    void set_sid(sid_t id) { sid = id; }
    std::string get_trace_id() const { return std::to_string(nid); }
    bool set_sid_if_null(sid_t id){
        if(sid.empty()) {
            sid=id;
            return 1;
        }
        return 0;
    }
    void set_trace_id(int id){
        nid = id;
    }
    virtual std::string represent() = 0;
    protected:
    std::string sid="";
    int64_t nid = -1;
};

template<typename obj>
obj& get_vec_elem_with_check(size_t n, std::vector<obj>& vec){
    auto vector_size = vec.size();
    CHECK_CONDITION(n<vector_size, "The query index exceeded the vector size.");
    return vec[n];
}

class value_impl_base {
    public:
    value_impl_base() = default;
};

class value : public graph_object{
public:
    value() = default;
    value(node * op);
    value(node *op, std::string sid);

    virtual ~value();
    virtual std::string represent(); 
    void print();
    void link_node(node* n){ users.push_back(n); }
    void dlink_node(node* n){
        auto iter = std::find(users.begin(), users.end(), n);
        if(iter!=users.end()) users.erase(iter);
    }
    void swap_node(node* from, node* to){
        auto iter = std::find(users.begin(), users.end(), from);
        if(iter!=users.end()) *iter = to;
    }
    void deprecate();

    std::vector<node*>& get_users(){ return users; }
    value_impl_base* get_impl() const { return impl; }
    void set_impl(value_impl_base* i){ impl = i; }
    void set_defining_op(node* op){ defop = op; }
    node* get_defining_op() const { return defop; }
    template<typename T>
    T* get_defining_op(){
        return dynamic_cast<T*>(defop);
    }

    void swap(value& v){
        // Note that this function doesn't swap impl!
        // swap the users
        std::swap(users, v.get_users());
        // swap the defining op
        auto buffer_op = defop;
        defop = v.get_defining_op();
        v.set_defining_op(buffer_op);
    }
    
    void remove_user(node* n);
    size_t get_user_size() const { return users.size(); }

    private:
    node* defop = nullptr;
    std::vector<node*> users;
    value_impl_base* impl = nullptr;
};

class dependencyValue : public value {
    public:
    dependencyValue() = default;
    dependencyValue(node * op) : value(op) { set_sid("dummy"); }
    std::string represent();
};

class node : public graph_object{
public : 
    // the first output value is dependency value used to store 
    // the dependency inform that don't have value connections
    node(std::string id="op", graph * g=nullptr) : 
    graph_object(id){ 
        graph_ = g;
    };
    virtual ~node() = default;
    virtual std::string represent() {
        printer p;
        p<<_v_.get()->represent()<<" = "<<get_sid() <<" : "<<represent_inputs();
        return p.dump();
    }
    virtual std::string represent_inputs();
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

    void replace_by(node* new_op);

    // register the input at the given position. Other inputs after 
    // that index will be push back by 1 pos.
    void register_input_at( value* val, size_t pos);

    template<typename...ARG>
    value* create_value(ARG...args){
        _v_ = std::make_unique<value>(args...);
        return _v_.get();
    }

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

    //void setTraceIDToOutput(context *ctx);
    int valueTraceIDStart = -1;

    bool isDependencyFullfilled();

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
        _vq.push(&entry);
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
            for(auto vn : val->getUsers()){
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
    LGFContext* get_context() { return ctx; }
    void set_context(LGFContext* c) { ctx = c; }
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
    LGFContext* ctx;
    // how many nodes contained in this graph
};

//----------------------------------------

}

#endif