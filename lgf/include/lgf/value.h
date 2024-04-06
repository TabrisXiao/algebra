#ifndef LGF_VALUE_H_
#define LGF_VALUE_H_
#include "object.h"
#include <string>
#include <vector>

namespace lgf{

class node;
class valueDesc : public graphObject{
    public:
    valueDesc() = default;
    valueDesc(sid_t id) : graphObject(id) {}
    virtual sid_t represent(){ return ""; }
};

class value : public graphObject{
    public:
    value() = default;
    value(node *op, std::string sid = "") : graphObject(sid), defop(op) {
    }
    value(valueDesc& d, node *op, std::string sid = "" ) : graphObject(sid), defop(op), desc(&d) {}

    virtual ~value();
    virtual sid_t represent(); 
    sid_t desc_represent(){
        if(desc) return desc->represent();
        return "";
    }
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
    valueDesc* get_desc() const { return desc; }
    void set_desc(valueDesc* i){ 
        desc = i; 
    }
    void consume(value& val){ 
        
    }
    void set_defining_op(node* op){ defop = op; }
    node* get_defining_op() const { return defop; }
    template<typename T>
    T* get_defining_op(){
        return dynamic_cast<T*>(defop);
    }

    void swap(value& v){
        // Note that this function doesn't swap desc!
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
    valueDesc* desc = nullptr;
};
} // namespace lgf

#endif