#ifndef LGF_GROUP_H
#define LGF_GROUP_H
#include <memory.h>
#include "operation.h"
#include "painter.h"
#include "pass.h"

namespace lgf{

class group {
    public:
    group() = default;
    virtual resultCode rewrite( painter, operation *op) = 0;
};

template<typename groupType>
class groupRewriter : public rewriterBase{
    public: 
    groupRewriter() = default;
    virtual resultCode execute( painter rewriter,operation* op) override final{
        if(auto g = dynamic_cast<groupType*>(op))
        {
            auto sig = g->rewrite(rewriter, op);
            return sig;
        }
        return resultCode::pass();
    }
};

class normalizer : public group {
    public: 
    normalizer() = default;
};

class normalizationPass : public passBase {
    public: 
    normalizationPass() : passBase("normalization"){ }
    virtual resultCode run(){
        painter p(getContext());
        addRewriter<groupRewriter<normalizer>>();
        // applyRewriterOnce(p, getGraph());
        // return applyRewriterOnce(p, getGraph());
        removeIdenticalOps(getGraph());
        resultCode code = applyRewriterGreedy(p, getGraph());
        inferTypes(getGraph());
        //removeUnusedOps(getGraph());
        //getGraph()->clean();
        return code;
    }

    operation* checkIfIdenticalExist(operation* op, std::queue<operation*> &list){
        std::queue<operation*> q=list;
        while(!q.empty()){
            auto checkop = q.front();
            q.pop();
            if(op->isIdentical(checkop)){
                return checkop;
            }
        }
        return nullptr;
    }

    void inferTypes(graph* g){
        auto ctx = g->getContext();
        for(auto op : g->getNodeList()){
            op->inferType( ctx );
            if(auto subg = dynamic_cast<graph*>(op)){
                inferTypes(subg);
            }
        }
    }

    void removeUnusedOps(graph* g){
        for(auto op : g->getNodeList()){
            bool canRemove = op->getStatus().isTrivial();
            for(auto & val : op->getOutputs()){
                if(val->getUserSize() !=0 ){
                    canRemove = false;
                    break;
                }
            }
            if(canRemove) op->erase();
            else if(auto subg = dynamic_cast<graph*>(op)){
                removeUnusedOps(subg);
            }
        }
    }

    bool removeIdenticalOps(graph* g){
        // using breadth first walk to remove identical ops
        bool changed = false;
        if(g == nullptr) {
            THROW("Remove identical ops failed: graph is invalid.");
        }
        auto list = g->getEntry().getIndirectDependecyValue()->getUsers();
        std::queue<operation*> queue;
        for(auto op : list){
            if(op->isRemovable() || !op->isActive()) continue;
            
            // if op is a graph:
            if(auto subg = dynamic_cast<graph*>(op)){
                
                removeIdenticalOps(subg);
                continue;
            }
            queue.push(op);
            // if( !checkIfIdenticalExist(op, queue) ){
            //     queue.push(op);
            // }
        }

        while(!queue.empty()){
            auto op = queue.front();
            queue.pop();
            for(auto& output: op->getOutputs()){
                // skip the dependencyValue
                if(auto dependency = dynamic_cast<dependencyValue*>(output.get())) 
                    continue;
                for(auto user : output->getUsers()){
                    if(user->isRemovable() || !user->isActive()) continue;
                     if(auto keepop = checkIfIdenticalExist(user, queue)){
                        user->replaceBy(keepop);
                        changed = true;
                    }else{
                        queue.push(user);
                    }
                }
            }
        }
        return changed;
    }
};

}

#endif