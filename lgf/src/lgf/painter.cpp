
#include "lgf/painter.h"
//#include "utility.h"
using namespace lgf;

bool painter::applyRewriterOnce(){
    bool ischanged = 0;
    for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++)
    {
        auto nodes = current_graph->getNodeList(); 
        for(auto & node : nodes){
            if(node->isRemovable() || !node->isActive()) continue;
            ischanged = (*ptr).get()->execute(*this, node) ||ischanged;
        }
        current_graph->clean();
    }
    current_graph->verify();
    return ischanged;
}
//---------------------------------------------------

int painter::applyRewriterGreedy(){
    bool repeat = applyRewriterOnce();
    current_graph->clean();
    int counts = 1;
    while(repeat){
        counts++;
        repeat = applyRewriterOnce();
        current_graph->clean();
    }
    return counts;
}

bool painter::walkApplyRewriterOnce(bool deepWalk){
    current_graph->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(*this, op);
        }
    }, 1, 1, 1, deepWalk);
    current_graph->clean();
    return 0;
}

bool painter::translation(){
    current_graph->walk([&](operation* op){
        if(op->isRemovable() || !op->isActive()) return;
        for(auto ptr=rewriters.begin(); ptr!=rewriters.end(); ptr++){
            (*ptr).get()->execute(*this, op);
        }
    }, 1, 1, 1, 0);
    current_graph->clean();
    return 0;
}