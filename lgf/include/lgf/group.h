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
class groupRewriter : public rewriterBase{
    public: 
    groupRewriter() = default;
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
        painter p(get_graph());
        add_rewriter<groupRewriter<normalizer>>();
        // applyRewriterOnce(p, getGraph());
        // return applyRewriterOnce(p, getGraph());
        remove_identical_ops(get_graph());
        resultCode code = apply_rewriter_greedy(p, get_graph());
        remove_unused_ops(get_graph());
        get_graph()->clean();
        return code;
    }

    node* check_if_identical_exist(node* op, std::queue<node*> &list){
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

    void remove_unused_ops(graph* g){
        for(auto op : g->get_nodes()){
            bool canRemove = op->is_trivial();
            auto val = op->output();
            if(val->get_user_size() !=0 ){
                canRemove = false;
                break;
            }
            if(canRemove) op->erase();
            else if(auto subg = dynamic_cast<graph*>(op)){
                remove_unused_ops(subg);
            }
        }
    }

    bool remove_identical_ops(graph* g){
        // using breadth first walk to remove identical ops
        // assignID is necessary for the checkIfIdenticalExist function as the id is used to check if two ops are identical
        g->assign_id(0);
        bool changed = false;
        if(g == nullptr) {
            THROW("Remove identical ops failed: graph is invalid.");
        }
        auto list = g->get_nodes();
        std::queue<node*> queue;
        for(auto op : list){
            if(op->is_removable() || !op->is_active()) continue;
            
            // if op is a graph:
            if(auto subg = dynamic_cast<graph*>(op)){
                
                remove_identical_ops(subg);
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
            auto output = op->output();
            for(auto user : output->get_users()){
                if(user->is_removable() || !user->is_active()) continue;
                 if(auto keepop = check_if_identical_exist(user, queue)){
                    user->replace_by(keepop);
                    changed = true;
                }else{
                    queue.push(user);
                }
            }
        }
        return changed;
    }
};

}

#endif