
#include "lgf/pass.h"
#include "lgf/group.h"
//#include "utility.h"
using namespace lgf;

bool passBase::applyRewriterOnce(painter &p, graph* g){
    bool ischanged = 0;
    p.gotoGraph(g);
    for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++)
    {
        auto nodes = g->getNodeList(); 
        for(auto & node : nodes){
            if(node->isRemovable() || !node->isActive()) continue;
            ischanged = (*ptr).get()->execute(p, node) ||ischanged;
            if(auto subg = dynamic_cast<graph*>(node)){
                applyRewriterOnce(p, subg);
            }
        }
    }
    g->verify();
    return ischanged;
}
//---------------------------------------------------

int passBase::applyRewriterGreedy(painter &p, graph* g){
    p.gotoGraph(g);
    bool repeat = applyRewriterOnce(p, g);
    g->clean();
    int counts = 1;
    while(repeat){
        counts++;
        repeat = applyRewriterOnce(p, g);
        g->clean();
    }
    return counts;
}

bool passBase::walkApplyRewriterOnce(painter &p, graph* g, bool deepWalk){
    p.gotoGraph(g);
    g->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(p, op);
        }
    }, 1, 1, 1, deepWalk);
    return 0;
}

bool passBase::translation(painter &p, graph* g){
    g->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(p, op);
        }
    }, 1, 1, 1, 0);
    return 0;
}

void passManager::addNormalizationPass(){
    addPass(std::make_unique<normalizationPass>());
}