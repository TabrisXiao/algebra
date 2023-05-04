
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include <unordered_set>
#include <memory.h>
#include "global.h"
#include "operation.h"

// abstract operation graph

namespace lgf{

class rewriterBase;

class painter {
    public : 
    painter(){ g = &origGraph; };
    painter(graph* reg_) { g = reg_; }
    ~painter(){}
    template<typename obj, typename...ARGS>
    obj* createOp(ARGS ...args){
        auto ptr = new obj(args...);
        g->addOp(ptr);
        return ptr;
    }

    // create a new op to replace the op1's users
    template<typename obj, typename...ARGS>
    obj* replaceOp(operation *op1, ARGS ...args){
        auto op2 = createOp<obj>(args...);
        op1->replaceBy(op2);
        op1->dropAllInputs();
        op1->setRemovable();
        return op2;
    }

    void erase(operation* op){
        op->dropAllInputs();
        auto& users = op->getOutgoings();
        for(auto user : users){
            auto& refs = user->getInputRefs();
            // removing all valueRef of users pointing to this op;
            for(auto iter=refs.begin(); iter!=refs.end();){
                if((*iter).getDefiningOp()== op ) 
                    iter = refs.erase(iter);
                else iter++;
            }
            // removing the this op from the users incoming list
            user->getIncomings().erase(op);
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
    
    void gotoGraph(graph * reg_) { g = reg_;}
    graph& getGraph(){ return *g;}
    void backToOrigGraph(){ g = &origGraph; }
    graph& getOrigGraph(){ return origGraph; }

    // addRewriter will create a rewriter using the arguments;
    template<typename T, typename ...ARGS>
    void addRewriter(ARGS...arg){ 
        auto ptr = std::make_unique<T>(arg...);
        rewriters.push_back(std::move(ptr));
    }
    bool applyRewriterOnce();
    int applyRewriterGreedy();

    private: 
    graph *g = nullptr;
    graph origGraph;
    std::vector<std::unique_ptr<rewriterBase>> rewriters;
};

class rewriterBase {
    public:
    rewriterBase() = default;
    virtual ~rewriterBase() = default;
    virtual int execute(painter &, operation * op) = 0;
};

template<typename concreteOp>
class rewriter : public rewriterBase{
    public : rewriter() = default;
    virtual bool rewrite(painter &, concreteOp *op) = 0;
    virtual int execute(painter & rewriter,operation* op) override final{
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