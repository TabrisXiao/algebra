
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include <unordered_set>
#include <memory.h>
#include <algorithm>
#include "global.h"
#include "node.h"
// abstract node graph

namespace lgf{

class rewriterBase;

class LGFContext;

class painter {
    public : 
    struct paintPoint{
        graph* g=nullptr;
        std::vector<node*>* nodes = nullptr;
        std::vector<node*>::iterator iter = std::vector<node*>::iterator();

        bool is_invalid(){
            if( g==nullptr || nodes==nullptr) return true;
            if( iter < nodes->begin() || iter > nodes->end())
                return true;
            return false;
        }
    };
    painter() = default;
    painter(graph * g) : point({g, &(g->get_nodes()), g->get_nodes().end()}), ctx(&(g->get_context())){}
    painter(painter &p)
    : point(p.getpaintPoint())
    , ctx(p.get_context()){}
    ~painter(){}
    void set_context(LGFContext * ctx_){
        ctx = ctx_;
    } 
    template<typename obj>
    obj* sketch(){
        auto op = obj::build();
        //op->inferType(ctx);
        return op;
    }
    template<typename obj, typename...ARGS>
    obj* sketch(ARGS ...args){
        auto op = obj::build(args...);
        //op->inferType(ctx);
        return op;
    }
    template<typename obj, typename...ARGS>
    obj* paint(ARGS ...args){
        THROW_WHEN(point.is_invalid(), "paint point is invalid!")
        //CHECK_CONDITION(point.g!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>(args...);
        //add to graph
        op->set_parent_graph(point.g);
        point.iter = point.nodes->insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }
    template<typename obj>
    obj* paint(){
        THROW_WHEN(point.is_invalid(), "paint point is invalid!")
        auto op = sketch<obj>();
        //add to graph
        op->set_parent_graph(point.g);
        point.iter = point.nodes->insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }

    template<typename origOp, typename targetOp>
    targetOp* isomorphic_rewrite(origOp* op){
        auto newop = new targetOp();
        newop->set_parent_graph(point.g);
        newop->register_inputs(op->get_inputs());

        auto & nodes = op->get_parent_graph()->get_nodes();
        auto iter = std::find(nodes.begin(), nodes.end(), op);
        *iter = newop;

        lastOp = newop;
        op->replaceBy(newop);
        return newop;
    }
    void set_paintPoint_to_top(){
        point.iter = point.g->get_nodes().begin();
    }
    void set_paintPoint_before(node* op){
        point.g = op->get_parent_graph();
        auto & vec = point.g->get_nodes();
        point.iter=std::find(vec.begin(), vec.end(),op);
        if(point.iter !=vec.begin()) point.iter--;
    }
    void set_paintPoint_after(node* op){
        point.g = op->get_parent_graph();
        auto & vec = point.g->get_nodes();
        point.iter=std::find(vec.begin(), vec.end(),op);
        if(point.iter !=vec.end()) point.iter++;
        else point.iter = point.iter-1;
    }

    void addOpToCurrentGraph(node* op){
        op->set_parent_graph(point.g);
        point.iter = point.g->get_nodes().insert(point.iter, op)+1;
    }

    // making an Op depends on the lastOp so that in a dependency walk order,
    // it will be later than the current lastOp
    void appendToCurrentGraph(node* op){
        op->set_parent_graph(point.g);
        point.iter = point.g->get_nodes().insert(point.iter, op)+1;
    }

    void replace_by(node *old, node *new_op){
        if(old->replace_by(new_op)){
            point.iter = point.nodes->end();
        }
        // don't remove the node is it is used by the new_op
        if(!old->is_user(new_op)){
            old->erase();
        }
    }

    // create a new op to replace the op1's users
    template<typename obj>
    obj* replace_op(node *op1){
        auto op2 = sketch<obj>();
        replace_by(op1, op2);
        return op2;
    }

    template<typename obj, typename...ARGS>
    obj* replace_op(node *op1, ARGS ...args){
        auto op2 = sketch<obj>(args...);
        replace_by(op1, op2);
        return op2;
    }
    
    // merge two ops:
    // op1   op2       op1 -- op2
    //   \   /          | \
    //     op    --->   |  \
    //    /  \         op3  op4
    //  op3  op4
    // Merge op into op1 will result to a new graph that op is removed
    // but the connection of op is inherited by op1
    void merge(node *op1, node * op2){
        // TODO
        return ;}
    
    void goto_graph(graph * reg_) {
        point.g = reg_;
        point.nodes = &(reg_->get_nodes());
        point.iter = point.nodes->end();
    }

    std::vector<value*>::iterator insert_inputs(std::vector<value*>::iterator target, std::vector<value*>::iterator begin, std::vector<value*>::iterator end, node* op){
        // insert the values between begin and end into the op's inputs at target position
        auto iter = op->get_inputs().insert(target, begin, end)+std::distance(begin, end);
        for(auto it = begin; it != end; it++){
            (*it)->link_node(op);
        }
        return iter;
    }

    // replace the op's input pointed by target by the its defining op's inputs (insert values at target position)
    std::vector<value*>::iterator replace_inputs(std::vector<value*>::iterator target, node *op){
        auto defop = (*target)->get_defining_op();
        op->replace_input(target, defop->input(0));
        target++;
        return insert_inputs(target, defop->get_inputs().begin()+1, defop->get_inputs().end(), op);
    }

    paintPoint getpaintPoint(){ return point; }
    
    graph* getGraph(){ return point.g;}
    
    void gotoParentGraph(){
        if(!point.g) return;
        goto_graph(point.g->get_parent_graph());
    }

    graph* get_parent_graph(){ 
        if(!point.g) return nullptr;
        return point.g->get_parent_graph(); 
    }

    LGFContext * get_context(){ return ctx; }
    paintPoint point;
    node * lastOp = nullptr;
    LGFContext* ctx = nullptr;
};

}

#endif