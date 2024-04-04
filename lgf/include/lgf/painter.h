
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

class painter {
    public : 
    struct paint_point{
        graph* g=nullptr;
        std::vector<node*>::iterator iter;
    };
    painter() = default;
    painter(LGFContext * ctx_) : ctx(ctx_) {}
    painter(painter &p)
    : point(p.getpaint_point())
    , ctx(p.get_context()){}
    ~painter(){}
    void setContext(LGFContext * ctx_){
        ctx = ctx_;
    } 
    template<typename obj>
    obj* sketch(){
        auto op = obj::build(ctx);
        //op->inferType(ctx);
        return op;
    }
    template<typename obj, typename...ARGS>
    obj* sketch(ARGS ...args){
        auto op = obj::build(ctx, args...);
        //op->inferType(ctx);
        return op;
    }
    template<typename obj, typename...ARGS>
    obj* paint(ARGS ...args){
        //CHECK_CONDITION(point.g!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>(args...);
        //add to graph
        op->set_parent_graph(point.g);
        point.iter = point.g->get_nodes().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }
    template<typename obj>
    obj* paint(){
        //CHECK_CONDITION(current_graph!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>();
        //add to graph
        op->set_parent_graph(point.g);
        point.iter = point.g->get_nodes().insert(point.iter, op)+1;
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
    void set_paint_point_to_top(){
        point.iter = point.g->get_nodes().begin();
    }
    void set_paint_point_before(node* op){
        point.g = op->get_parent_graph();
        auto & vec = point.g->get_nodes();
        point.iter=std::find(vec.begin(), vec.end(),op);
        if(point.iter !=vec.begin()) point.iter--;
    }
    void set_paint_point_after(node* op){
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

    // create a new op to replace the op1's users
    template<typename obj>
    obj* replace_op(node *op1){
        auto op2 = sketch<obj>();
        op1->replace_by(op2);
        auto & nodes = op1->get_parent_graph()->get_nodes();
        // find the op1 in nodes and assign it with the op2
        auto iter = std::find(nodes.begin(), nodes.end(), op1);
        *iter = op2;
        op2->set_parent_graph(op1->get_parent_graph());
        op1->erase();
        return op2;
    }

    template<typename obj, typename...ARGS>
    obj* replace_op(node *op1, ARGS ...args){
        auto op2 = sketch<obj>(args...);
        op1->replace_by(op2);
        auto & nodes = op1->get_parent_graph()->get_nodes();
        // find the op1 in nodes and assign it with the op2
        auto iter = std::find(nodes.begin(), nodes.end(), op1);
        *iter = op2;

        op2->set_parent_graph(op1->get_parent_graph());
        op1->erase();
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
        point.iter = reg_->get_nodes().end();
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

    paint_point getpaint_point(){ return point; }
    
    graph* getGraph(){ return point.g;}
    void gotoParentGraph(){
        if(!point.g) return;
        goto_graph(point.g->get_parent_graph());
    }
    graph* get_parent_graph(){ 
        if(!point.g) return nullptr;
        return point.g->get_parent_graph(); }
    LGFContext * get_context(){ return ctx; }
    paint_point point;
    node * lastOp = nullptr;
    LGFContext *ctx = nullptr;
};



}

#endif