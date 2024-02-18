
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include <unordered_set>
#include <memory.h>
#include <algorithm>
#include "global.h"
#include "operation.h"

// abstract operation graph

namespace lgf{

class rewriterBase;

class painter {
    public : 
    struct paintPoint{
        graph* g=nullptr;
        std::vector<operation*>::iterator iter;
    };
    painter(LGFContext * ctx_) : ctx(ctx_) {}
    painter(painter &p)
    : point(p.getPaintPoint())
    , ctx(p.getContext()){}
    ~painter(){}
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
        op->setParentGraph(point.g);
        if(op->getInputSize() == 0)
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }
    template<typename obj>
    obj* paint(){
        //CHECK_CONDITION(current_graph!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>();
        //add to graph
        op->setParentGraph(point.g);
        if(op->getInputSize() == 0) 
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }
    template<typename obj, typename...ARGS>
    obj* paintNoAppend(ARGS ...args){
        //CHECK_CONDITION(point.g!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>(args...);
        //add to graph
        op->setParentGraph(point.g);
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }
    template<typename obj>
    obj* paintNoAppend(){
        //CHECK_CONDITION(current_graph!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>();
        //add to graph
        op->setParentGraph(point.g);
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }

    template<typename origOp, typename targetOp>
    targetOp* isomorphicRewrite(origOp* op){
        auto newop = new targetOp();
        newop->setParentGraph(point.g);
        if(op->getInputSize() !=0 ) 
            newop->registerInputs(op->getInputs());
        else {
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        }

        if(op->getOutputSize() > 1){
            auto value = op->outputValue(1);
            newop->createValue(value->getType(), value->getSID());
        }

        auto & nodes = op->getParentGraph()->getNodeList();
        auto iter = std::find(nodes.begin(), nodes.end(), op);
        *iter = newop;

        lastOp = newop;
        op->replaceBy(newop);
        return newop;
    }

    void setPaintPointBefore(operation* op){
        point.g = op->getParentGraph();
        auto & vec = point.g->getNodeList();
        point.iter=std::find(vec.begin(), vec.end(),op);
        if(point.iter !=vec.begin()) point.iter--;
    }
    void setPaintPointAfter(operation* op){
        point.g = op->getParentGraph();
        auto & vec = point.g->getNodeList();
        point.iter=std::find(vec.begin(), vec.end(),op);
        if(point.iter !=vec.end()) point.iter++;
        else point.iter = point.iter-1;
    }

    void addOpToCurrentGraph(operation* op){
        op->setParentGraph(point.g);
        if(op->getInputSize() == 0)
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
    }

    // making an Op depends on the lastOp so that in a dependency walk order, 
    // it will be later than the current lastOp
    void appendToCurrentGraph(operation* op){
        op->setParentGraph(point.g);
        if(point.g->getNodeList().begin() != point.iter) op->dependOn(*(point.iter-1));
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
    }

    // create a new op to replace the op1's users
    template<typename obj>
    obj* replaceOp(operation *op1){
        auto op2 = sketch<obj>();
        op1->dropAllInputs();
        for(auto i=0; i<op1->getOutputSize(); i++){
            op1->outputValue(i)->swap(op2->outputValue(i));
        }
        
        auto & nodes = op1->getParentGraph()->getNodeList();
        // find the op1 in nodes and assign it with the op2
        auto iter = std::find(nodes.begin(), nodes.end(), op1);
        *iter = op2;
        
        op2->setParentGraph(op1->getParentGraph());
        op1->erase();
        return op2;
    }
    template<typename obj, typename...ARGS>
    obj* replaceOp(operation *op1, ARGS ...args){
        auto op2 = sketch<obj>(args...);
        op1->dropAllInputs();
        for(auto i=0; i<op1->getOutputSize(); i++){
            op1->outputValue(i)->swap(op2->outputValue(i));
        }
        auto & nodes = op1->getParentGraph()->getNodeList();
        // find the op1 in nodes and assign it with the op2
        auto iter = std::find(nodes.begin(), nodes.end(), op1);
        *iter = op2;

        op2->setParentGraph(op1->getParentGraph());
        op1->erase();
        return op2;
    }


    void erase(operation* op){
        op->dropAllInputs();
        // drop users of output values from this op
        for(auto i = 0; i<op->getOutputSize(); i++){
            op->outputValue(i)->disconnectOp(op);
        }
        op->setRemovable();
    }
    
    // merge two ops:
    // op1   op2       op1 -- op2
    //   \   /          | \
    //     op    --->   |  \
    //    /  \         op3  op4
    //  op3  op4
    // Merge op into op1 will result to a new graph that op is removed
    // but the connection of op is inherited by op1
    void merge(operation *op1, operation * op2){
        // TODO
        return ;}
    
    void gotoGraph(graph * reg_) {
        point.g = reg_;
        point.iter = reg_->getNodeList().end();
    }

    std::vector<value*>::iterator insertValuesAsOpInputs(std::vector<value*>::iterator target, std::vector<value*>::iterator begin, std::vector<value*>::iterator end, operation* op){
        // insert the values between begin and end into the op's inputs at target position
        auto iter = op->getInputs().insert(target, begin, end)+std::distance(begin, end);
        for(auto it = begin; it != end; it++){
            (*it)->addUser(op);
        }
        return iter;
    }

    // replace the op's input pointed by target by the its defining op's inputs (insert values at target position)
    std::vector<value*>::iterator replaceInputByDefOpInputs(std::vector<value*>::iterator target, operation *op){
        auto defop = (*target)->getDefiningOp();
        op->replaceInputValueBy(target, defop->inputValue(0));
        target++;
        return insertValuesAsOpInputs(target, defop->getInputs().begin()+1, defop->getInputs().end(), op);
    }

    paintPoint getPaintPoint(){ return point; }
    
    graph* getGraph(){ return point.g;}
    void gotoParentGraph(){
        if(!point.g) return;
        gotoGraph(point.g->getParentGraph());
    }
    graph* getParentGraph(){ 
        if(!point.g) return nullptr;
        return point.g->getParentGraph(); }
    LGFContext * getContext(){ return ctx; }
    paintPoint point;
    operation * lastOp = nullptr;
    LGFContext *ctx = nullptr;
};



}

#endif