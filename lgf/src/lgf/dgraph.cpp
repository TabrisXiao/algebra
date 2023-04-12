
#include <stdarg.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <queue>
#include "lgf/dgraph.h"

// directed graph
namespace dgl
{
    void dedge::connect(vertex* v1, vertex *v2){
        if(vertexFrom){
            CHECK_VALUE(vertexFrom,v1,"This new connection attaches from a different vertex than original, which is not allowed.");
        }
        else {
            v1->attachTo(this);
        }
        v2->attachFrom(this);
    }
    void dedge::eraseVertexTo(vertex* v){
        for(auto iter = verticesTo.begin(); iter!=verticesTo.end();){
            if(v== *iter){
                iter=verticesTo.erase(iter);
                return;
            } else iter++;
        }
    }
    void dedge::dropOutgoingConnection(vertex *v){
        eraseVertexTo(v);
        v->eraseInEdge(this);
    }
    void dedge::dropAllOutgoingConnection(){
        std::cout<<"droping"<<std::endl;
        for(auto & vtx: verticesTo){
            vtx->eraseInEdge(this);
        }
        verticesTo.clear();
    }
    void vertex::eraseInEdge(dedge* e){
        // assuming there's at most one edge needs to be erased
        // ie. no duplication connections.
        std::cout<<"before "<<inEdges.size()<<std::endl;
        auto iter = std::find(inEdges.begin(), inEdges.end(), e);
        if(iter!=inEdges.end()) inEdges.erase(iter);
        for(auto iter = inEdges.begin(); iter!=inEdges.end(); iter++){
            std::cout<<e<<" : "<<*iter<<std::endl;
            if(e==*iter){
                inEdges.erase(iter);
                return;
            }
        }
        std::cout<<"after "<<inEdges.size()<<std::endl;
    }
    void vertex::eraseOutEdge(dedge* e){
        // assuming there's at most one edge needs to be erased
        // ie. no duplication connections.
        for(auto iter = outEdges.begin(); iter!=outEdges.end(); iter++){
            if(e==*iter){
                outEdges.erase(iter);
                return;
            }
        }
    }
    // void vertex::detachInput(dedge *e){
    //     auto iter = std::find(inEdges.begin(), inEdges.end(), e);
    //     while(iter!=inEdges.end()){
    //         inEdges.erase(iter);
    //         iter = std::find(inEdges.begin(), inEdges.end(), e);
    //     }
    //     e->resetVerticesTo();
    // }
    // void vertex::detachOutput(dedge *e){
    //     auto iter = std::find(outEdges.begin(), outEdges.end(), e);
    //     while(iter!=outEdges.end()){
    //         outEdges.erase(iter);
    //         iter = std::find(outEdges.begin(), outEdges.end(), e);
    //     }
    //     e->resetVertexFrom();
    // }
    void vertex::detach()
    {
        for (auto& e : inEdges)
        {
            e->eraseVertexTo(this);
        }
        inEdges.clear();
        for (auto e : outEdges)
        {
            e->eraseVertexFrom(this);
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

} // namespace dgl
