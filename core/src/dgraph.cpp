
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
    void dedge::connect(vertex* v1, vertex *v2){
        if(vertexFrom){
            CHECK_VALUE(vertexFrom,v1);
        }
        else {
            attachFrom(v1);
            v1->attachTo(this);
        }
        attachTo(v2);
        v2->attachFrom(this);
    }
    void vertex::detachInput(dedge *e){
        auto iter = std::find(inEdges.begin(), inEdges.end(), e);
        while(iter!=inEdges.end()){
            inEdges.erase(iter);
            iter = std::find(inEdges.begin(), inEdges.end(), e);
        }
        e->resetVerticesTo();
    }
    void vertex::detachOutput(dedge *e){
        auto iter = std::find(outEdges.begin(), outEdges.end(), e);
        while(iter!=outEdges.end()){
            outEdges.erase(iter);
            iter = std::find(outEdges.begin(), outEdges.end(), e);
        }
        e->resetVertexFrom();
    }
    void vertex::detach()
    {
        for (auto e : inEdges)
        {
            auto vtx = e->getVertexFrom();
            vtx->detachOutput(e);
        }
        inEdges.clear();
        for (auto e : outEdges)
        {
            for(auto ver : e->getVerticesTo())
                ver->detachInput(e);
        }
        outEdges.clear();
    }

    // check if all the outgoing vertice have been marked as explored
    // bool vertex::isExploredOugoingVertex()
    // {
    //     for (auto e : getOutDedges())
    //     {
    //         if (!(e->getReceiver()->isExplored()))
    //             return 0;
    //     }
    //     return 1;
    // }
    // // check if all the incoming vertice have been marked as explored
    // bool vertex::isExploredIncomingVertex()
    // {
    //     for (auto e : getInEdges())
    //     {
    //         if (!(e->getSender()->isExplored()))
    //             return 0;
    //     }
    //     return 1;
    // }

    //template <typename callable>
    //void graph::BFWalk(callable &&fn){
        

} // namespace dgl
