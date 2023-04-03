
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include <unordered_set>
#include <memory.h>
#include "sdgraph.h"
#include "global.h"
#include "operation.h"

// abstract operation graph

namespace aog{

class opModifier {
    public : 
    opModifier() = default;
    template<typename obj, typename...ARGS>
    obj* create(ARGS ...args){
        auto ptr = new obj(args...);
        if(!(ptr->hasInput())){
            reg->addL1Vertex(dynamic_cast<sdgl::vertex*>(ptr));
        }
        return ptr;
    }
    void deactiveOp(operation *op){
        op->detach();
        op->setActivation(0);
        _removed_ops.push_back(op);
    }
    void removeElement(element * e);
    void removeOp(operation *op){
        // remove the output element from the users
        auto & vertices = op->getOutVertices();
        for(auto vert : vertices){
            auto user = dynamic_cast<operation*>(vert);
            auto & inputs = user->getInputs();
            auto iter = inputs.begin();
            while (iter!=inputs.end()){
                if((*iter)->getDefiningOp() == op){
                    inputs.erase(iter);
                    iter = inputs.begin();
                } else { iter++; }
            }
        }
        // detach the operation from the graph
        // since the op is removed, the inputs keep in op is unchanged.
        // But this won't affect the rest of the graph so we don't need
        // to remove the input elements
        deactiveOp(op);
    }
    void replaceOperation(operation *origOp, operation *newOp){
        auto & out_ops = origOp->getOutVertices();
        for(auto op_ : out_ops){
            newOp->linkTo(*op_);
        }
        auto & in_ops = origOp->getInVertices();
        for(auto op_ : in_ops){
            newOp->linkFrom(*op_);
        }
    }
    // replace the element by the new element for all users
    void replaceElement(element *e, element* newe){
        auto vec = e->getUsers();
        for(auto op : vec){
            auto & inputs = op->getInputs();
            for(auto i=0 ; i<inputs.size(); i++){
                if(inputs[i]==e){
                    inputs[i]=newe;
                    break;
                }
            }
        }
    }
    // create a new type T op and replace the origOp.
    // for both operations has to have only one elements;
    template<typename T, typename...ARGS>
    T* replaceOp(operation *origOp, ARGS ...args){
        auto op = create<T>(args...);
        // the input vertices is linked when creating the op, 
        // so we only need to link the output vertices
        replaceElement(&(origOp->getOutput(0)), &(op->getOutput(0)));
        replaceOperation(origOp, op);
        deactiveOp(origOp);
        return op;
    }

    void replaceOp(operation *origOp, operation *newOp){
        replaceElement(&(origOp->getOutput(0)), &(newOp->getOutput(0)));
        replaceOperation(origOp, newOp);
        deactiveOp(origOp);
    }
    
    void flush(){
        for(auto op : _removed_ops){
            delete op;
        }
        _removed_ops.clear();
    }
    void setWorkRegion(region * reg_) { reg = reg_;}
    region *reg = nullptr;
    std::vector<operation* > _removed_ops;
};

class opBuilder : public opModifier{
    class regionPtrHelper{
    public : 
        regionPtrHelper(opBuilder *builder) : ptr(builder){}
        ~regionPtrHelper(){ 
        }
        region *currentRegion=nullptr, *previousRegion=nullptr;
        opBuilder* ptr= nullptr;
    };
public:
    opBuilder(context *ctx_) {
        ctx = ctx_;
        setWorkRegion(ctx->getRegion());
    }
   
    context * getContext(){return ctx;}
    context *ctx=nullptr;
};

class opRewriter : public opModifier{
    public : 
    opRewriter() = default;
    opRewriter(context *ctx_){
        ctx = ctx_;
        CHECK_CONDITION(ctx!=nullptr)
        setWorkRegion(ctx->getRegion());
    }
    context * ctx = nullptr;
};

class rewriterBase {
    public:
    rewriterBase() = default;
    virtual ~rewriterBase() = default;
    virtual int execute(opRewriter &, operation * op) = 0;
};

template<typename concreteOp>
class rewriter : public rewriterBase{
    public : rewriter() = default;
    virtual bool rewrite(opRewriter &, concreteOp *op) = 0;
    virtual int execute(opRewriter & rewriter,operation* op) override final{
        // rewrite return value: 
        // 1 rewrite happen
        // 0 rewrite failed or not matched
        if(auto cop = dynamic_cast<concreteOp*>(op))
        {
            return int(rewrite(rewriter, cop));
        }
        return 0;
    }
};

class graphModifier {
    public: 
    graphModifier() = default;
    template<typename T, typename ...ARGS>
    void addRewriter(ARGS...arg){ 
        auto ptr = std::make_unique<T>(arg...);
        rewriters.push_back(std::move(ptr));
    }
    void addRewriter(std::unique_ptr<rewriterBase> rewriter){
        rewriters.push_back(std::move(rewriter));
    }
    // apply the rewriters through BFWalk, 
    // Each rewriter is applied only once
    bool walkApplyOnce(region* reg){
        bool ischanged = 0;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++)
        {
            reg->getEntryVertex().BFWalk([&](sdgl::vertex* _op){
                if(auto op = dynamic_cast<operation*>(_op)){
                    ischanged = ischanged || (*ptr).get()->execute(_rw, op);
                }
            });
        }
        return ischanged;
    }
    
    // return how many run it repeated;
    int walkApplyGreedy(region* reg){
        bool repeat = repeat = walkApplyOnce(reg);
        int counts = 1;
        while(repeat){
            counts++;
            repeat = walkApplyOnce(reg);
        }
        return counts;
    }
    std::vector<std::unique_ptr<rewriterBase>> rewriters;
    opRewriter _rw;
};

}

#endif