
#include "lgf/pass.h"
#include "lgf/group.h"
//#include "utility.h"
using namespace lgf;

resultCode passBase::applyRewriterOnce(painter &p, graph* g){
    resultCode result;
    p.gotoGraph(g);
    for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++)
    {
        auto nodes = g->getNodeList(); 
        for(auto & node : nodes){
            if(node->isRemovable() || !node->isActive()) continue;
            result.add((*ptr).get()->execute(p, node));
            if(auto subg = dynamic_cast<graph*>(node)){
                result.add(applyRewriterOnce(p, subg));
            }
        }
    }
    return result;
}
//---------------------------------------------------

resultCode passBase::applyRewriterGreedy(painter &p, graph* g){
    p.gotoGraph(g);
    auto result = applyRewriterOnce(p, g);
    g->clean();
    int counts = 1;
    auto final_result = result;
    while(result.isSuccess()){
        counts++;
        result = applyRewriterOnce(p, g);
        final_result.add(result);
        g->clean();
    }
    return final_result;
}
//---------------------------------------------------

resultCode passBase::walkApplyRewriterOnce(painter &p, graph* g, bool deepWalk){
    p.gotoGraph(g);
    resultCode result;
    g->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            result.add((*ptr).get()->execute(p, op));
        }
    }, 1, 1, 1, deepWalk);
    return result;
}
//---------------------------------------------------

bool passBase::translation(painter &p, graph* g){
    g->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(p, op);
        }
    }, 1, 1, 1, 0);
    return 0;
}
//---------------------------------------------------

void passManager::validation(graph* g){
    auto nodes = g->getNodeList();
    for(auto & node : nodes){
        if(node->isRemovable() || !node->isActive()) continue;
        if(auto subg = dynamic_cast<graph*>(node)){
            validation(subg);
        } 
        node->validation();
        if(node->getStatus().isTrivial()) redundantCheck(node);
    }
    if(g->clean()){
        validation(g);
    }
}
//---------------------------------------------------

void passManager::addNormalizationPass(){
    addPass(std::make_unique<normalizationPass>());
}
//---------------------------------------------------