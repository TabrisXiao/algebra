
#include "lgf/pass.h"
#include "lgf/group.h"
//#include "utility.h"
using namespace lgf;

resultCode passBase::apply_rewriter_once( graph* g ){
    resultCode result;
    p.goto_graph(g);
    for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++)
    {
        auto nodes = g->get_nodes(); 
        for(auto & node : nodes){
            if(node->is_removable() || !node->is_active()) continue;
            result.add((*ptr).get()->execute(p, node));
            if(auto subg = dynamic_cast<graph*>(node)){
                result.add(apply_rewriter_once(subg));
            }
        }
    }
    return result;
}
//---------------------------------------------------

resultCode passBase::apply_rewriter_greedy( graph* g ){
    p.goto_graph(g);
    auto result = apply_rewriter_once(g);
    g->clean();
    int counts = 1;
    auto final_result = result;
    while(result.isSuccess()){
        counts++;
        result = apply_rewriter_once(g);
        final_result.add(result);
        g->clean();
    }
    return final_result;
}
//---------------------------------------------------

resultCode passBase::walk_apply_rewriter_once(painter &p, graph* g, bool deepWalk){
    p.goto_graph(g);
    resultCode result;
    g->walk([&](node* op){
        if(op->is_removable() || !op->is_active()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            result.add((*ptr).get()->execute(p, op));
        }
    }, deepWalk);
    return result;
}
//---------------------------------------------------

bool passBase::translation(painter &p, graph* g){
    g->walk([&](node* op){
        if(op->is_removable() || !op->is_active()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(p, op);
        }
    }, 0);
    return 0;
}
//---------------------------------------------------

void passManager::validation(graph* g){
    auto nodes = g->get_nodes();
    for(auto & node : nodes){
        if(node->is_removable() || !node->is_active()) continue;
        if(auto subg = dynamic_cast<graph*>(node)){
            validation(subg);
        } 
    }
    if(g->clean()){
        validation(g);
    }
}
//---------------------------------------------------

void passManager::add_normalization_pass(){
    add_pass(std::make_unique<normalizationPass>());
}
//---------------------------------------------------