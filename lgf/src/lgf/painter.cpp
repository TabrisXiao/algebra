
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