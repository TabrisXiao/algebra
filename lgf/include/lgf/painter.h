
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
        return obj::build(ctx);
    }
    template<typename obj, typename...ARGS>
    obj* sketch(ARGS ...args){
        return obj::build(ctx, args...);
    }
    template<typename obj, typename...ARGS>
    obj* paint(ARGS ...args){
        CHECK_CONDITION(point.g!=nullptr, "No graph associated to the painter!");
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
        CHECK_CONDITION(current_graph!=nullptr, "No graph associated to the painter!");
        auto op = sketch<obj>();
        //add to graph
        op->setParentGraph(point.g);
        if(op->getInputSize() == 0) 
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        point.iter = point.g->getNodeList().insert(point.iter, op)+1;
        lastOp = op;
        return op;
    }

    void setPaintPointAt(operation* op){
        point.g = op->getParentGraph();
        auto & vec = point.g->getNodeList();
        point.iter=std::find(vec.begin(), vec.end(),op);
    }

    void addOpAtCurrentPoint(operation* op){
        op->setParentGraph(point.g);
        if(op->getInputSize() == 0) 
            op->appendTo(dynamic_cast<operation*>(&(point.g->getEntry())));
        point.iter = point.g->getNodeList().insert(point.iter, op);
    }

    // making an Op depends on the lastOp so that in a dependency walk order, 
    // it will be later than the current lastOp
    void appendOp(operation* op){
        if(auto g = dynamic_cast<graph*>(lastOp)){
            op->dependOn(&(g->getEntry()));
        } else{
            op->dependOn(lastOp);
        }
        lastOp = op;
    }

    // create a new op to replace the op1's users
    template<typename obj>
    obj* replaceOp(operation *op1){
        auto op2 = sketch<obj>();
        op1->dropAllInputs();
        for(auto i=0; i<op1->getOutputSize(); i++){
            op1->outputValue(i)->replaceBy(op2->outputValue(i));
        }
        
        auto & nodes = op1->getParentGraph()->getNodeList();
        //std::replace(nodes.begin(), nodes.end(), op1, op2);
        for(auto & node : nodes) {
            if(node == op1) {
                node = op2;
            }
        }
        op2->setParentGraph(op1->getParentGraph());
        op1->setRemovable();
        return op2;
    }
    template<typename obj, typename...ARGS>
    obj* replaceOp(operation *op1, ARGS ...args){
        auto op2 = sketch<obj>(args...);
        op1->dropAllInputs();
        for(auto i=0; i<op1->getOutputSize(); i++){
            op1->outputValue(i)->replaceBy(op2->outputValue(i));
        }
        auto & nodes = op1->getParentGraph()->getNodeList();
        //std::replace(nodes.begin(), nodes.end(), op1, op2);
        for(auto & node : nodes) {
            if(node == op1) {
                node = op2;
            }
        }
        op2->setParentGraph(op1->getParentGraph());
        op1->setRemovable();
        return op2;
    }

    void erase(operation* op){
        op->dropAllInputs();
        // drop users of output values from this op
        for(auto i = 0; i<op->getOutputSize(); i++){
            op->outputValue(i)->dropUser(op);
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

class rewriterBase {
    public:
    rewriterBase() = default;
    virtual ~rewriterBase() = default;
    virtual int execute( painter, operation * op) = 0;
    LGFContext* getContext(){ return ctx; }
    LGFContext *ctx = nullptr;
};

template<typename concreteOp>
class rewriter : public rewriterBase{
    public : rewriter() = default;
    virtual bool rewrite( painter, concreteOp *op) = 0;
    virtual int execute( painter rewriter,operation* op) override final{
        // rewrite return value: 
        // 1 rewrite happen
        // 0 rewrite failed or not matched
        if(auto cop = dynamic_cast<concreteOp*>(op))
        {
            auto sig = rewrite(rewriter, cop);
            return int(sig);
        }
        return 0;
    }
};


}

#endif