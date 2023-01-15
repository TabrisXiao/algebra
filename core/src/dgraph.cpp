
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

    void vertex::detach()
    {
        for (auto e : inEdges)
        {
            e->vertexTo = nullptr;
        }
        inEdges.clear();
        for (auto e : outEdges)
        {
            e->vertexFrom = nullptr;
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
