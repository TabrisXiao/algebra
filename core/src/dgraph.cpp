
#include <stdarg.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <queue>
#include "dgraph.h"

// directed graph
namespace dgl
{
    void edge::connect(vertex &vfrom, vertex &vto)
    {
        attachTo(vto);
        attachFrom(vfrom);
        vto.attachFrom(*this);
        vfrom.attachTo(*this);
    }
    void vertex::detachInput(edge *e){
        auto iter = std::find(inEdges.begin(), inEdges.end(), e);
        while(iter!=inEdges.end()){
            inEdges.erase(iter);
            iter = std::find(inEdges.begin(), inEdges.end(), e);
        }
        e->resetReceiver();
    }
    void vertex::detachOutput(edge *e){
        auto iter = std::find(outEdges.begin(), outEdges.end(), e);
        while(iter!=outEdges.end()){
            outEdges.erase(iter);
            iter = std::find(outEdges.begin(), outEdges.end(), e);
        }
        e->resetSender();
    }
    void vertex::detach()
    {
        for (auto e : inEdges)
        {
            auto vtx = e->getSender();
            vtx->detachOutput(e);
        }
        inEdges.clear();
        for (auto e : outEdges)
        {
            auto vtx = e->getReceiver();
            vtx->detachInput(e);
        }
        outEdges.clear();
    }

    // check if all the outgoing vertice have been marked as explored
    bool vertex::isExploredOugoingVertex()
    {
        for (auto e : getOutEdges())
        {
            if (!(e->getReceiver()->isExplored()))
                return 0;
        }
        return 1;
    }
    // check if all the incoming vertice have been marked as explored
    bool vertex::isExploredIncomingVertex()
    {
        for (auto e : getInEdges())
        {
            if (!(e->getSender()->isExplored()))
                return 0;
        }
        return 1;
    }

    //template <typename callable>
    //void graph::BFWalk(callable &&fn){
        

} // namespace dgl
