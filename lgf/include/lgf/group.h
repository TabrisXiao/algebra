#ifndef LGF_GROUP_H
#define LGF_GROUP_H
#include <memory.h>
#include "node.h"
#include "painter.h"
#include "pass.h"

namespace lgf{

class group {
    public:
    group() = default;
    virtual resultCode rewrite( painter, node *op) = 0;
};

template<typename groupType>
class group_rewriter : public rewriter_base{
    public: 
    group_rewriter() = default;
    virtual resultCode execute( painter rewriter,node* op) override final{
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
        painter p(get_context());
        add_rewriter<group_rewriter<normalizer>>();
        // applyRewriterOnce(p, getGraph());
        // return applyRewriterOnce(p, getGraph());
        removeIdenticalOps(get_graph());
        resultCode code = apply_rewriter_greedy(p, get_graph());
        inferTypes(get_graph());
        removeUnusedOps(get_graph());
        get_graph()->clean();
        return code;
    }

    node* checkIfIdenticalExist(node* op, std::queue<node*> &list){
        std::queue<node*> q=list;
        while(!q.empty()){
            auto checkop = q.front();
            q.pop();
            if(op->is_identical(checkop)){
                return checkop;
            }
        }
        return nullptr;
    }

    void inferTypes(graph* g){
        auto ctx = g->get_context();
        for(auto op : g->get_nodes()){
            if(auto subg = dynamic_cast<graph*>(op)){
                inferTypes(subg);
            }
        }
    }

    void removeUnusedOps(graph* g){
        for(auto op : g->get_nodes()){
            bool canRemove = op->isTrivial();
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
        // assignID is necessary for the checkIfIdenticalExist function as the id is used to check if two ops are identical
        g->assign_id(0);
        bool changed = false;
        if(g == nullptr) {
            THROW("Remove identical ops failed: graph is invalid.");
        }
        auto list = g->getEntry().getIndirectDependecyValue()->getUsers();
        std::queue<node*> queue;
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