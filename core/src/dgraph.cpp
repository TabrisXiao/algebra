
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

    template <typename callable>
    void graph::BFWalk(callable &fn){
        //The breadth-first walk through the entries of graph.
        //call the callable at the begining of visiting each vertex.
        //the callable should return void.
        //the fn will be executed on each vertex at most once.
        std::vector<vertex *> vertice_buffer;
        std::queue<vertex *> _vq;
        for(auto v : subvertices){
            _vq.push(v);
        }
        while (_vq.size())
            {
                auto v = _vq.front();
                _vq.pop();
                fn(v);
                v->bExplored = 1;
                vertice_buffer.push_back(v);
                auto edges = v->getOutEdges();
                for (auto &e : edges)
                {
                    if (auto vn = e->getReceiver())
                    {
                        if (!(vn->bExplored))
                        {
                            _vq.push(vn);
                        }
                    }
                }
            }
            for (auto &v : vertice_buffer)
            {
                v->reset();
            }
        return; 
    }

} // namespace dgl
