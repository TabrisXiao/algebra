
#ifndef SDGRAPH_H
#define SDGRAPH_H
#include <vector>
#include <iostream>
#include <queue>
// Simplified Directed Graph Library
// Only vertex is used to construct graphs. The edge is represented by connection
// between vertices.
namespace sdgl{
class vertex{
public : 
    vertex () =default;
    virtual ~vertex() {}
    vertex(vertex &rhs)
    {
        bExplored = rhs.bExplored;
    }
    vertex(vertex &&rhs) : outVertices(std::move(rhs.getOutput())),
                           inVertices(std::move(rhs.getInput())),
                           bExplored(rhs.bExplored)
    {
        rhs.~vertex();
    }
    std::vector<vertex*> & getInput(){return inVertices;}
    std::vector<vertex*> & getOutput(){return outVertices;}
    //
    // attach this vertex to/from other vertices
    //
    template <typename... ARGS>
    void addInputVertex(ARGS &...args)
    {
        auto vertices = {&args...};
        for (auto v : vertices)
        {
            auto ptr = dynamic_cast<vertex*>(v);
            inVertices.push_back(ptr);
            ptr->addOutputVertex(*this);
        }
    }
    template <typename... ARGS>
    void addOutputVertex(ARGS &...args)
    {
        auto vertices = {&args...};
        for (auto v : vertices)
        {
            auto ptr = dynamic_cast<vertex*>(v);
            outVertices.push_back(ptr);
            ptr->addInputVertex(*this);
        }
    }
    bool hasInput (){return inVertices.size()!=0;}
    bool hasOutput(){return outVertices.size()!=0;}
    void reset() { bExplored = 0; }
    template <typename callable>
    void BFWalk(callable &&fn){
        //The breadth-first walk through the graph starting at this vertex.
        //call the callable at the begining of visiting each vertex.
        //the callable should return void.
        //the fn will be executed on each vertex at most once.
        std::vector<vertex *> vertice_buffer;
        std::queue<vertex *> _vq;
        _vq.push(this);
        while (_vq.size())
            {
                auto v = _vq.front();
                _vq.pop();
                vertice_buffer.push_back(v);
                auto vertices = v->getOutput();
                for (auto &vn : vertices)
                {
                    if (!(vn->bExplored))
                    {
                        _vq.push(vn);
                    }
                }
                fn(v);
                v->bExplored = 1;
            }
            for (auto &v : vertice_buffer)
            {
                v->reset();
            }
        return;
    }

    bool bExplored = 0;
    std::vector<vertex*> outVertices, inVertices;
};

class sdgraph {
    public : 
    sdgraph () = default;
    virtual ~sdgraph(){
        entry.BFWalk([](vertex* vtx){
            delete vtx;
        });
    }
    void addTopLevelVertex(vertex* vtx){
        entry.addOutputVertex(*vtx);
    }
    vertex & getEntry(){return entry;}
    vertex & getReturn(){return exit;}
    vertex & addVertex(){
        auto v = new vertex();
        entry.addOutputVertex(*v);
    }
    template <typename... ARGS>
    vertex & addVertex(ARGS&...args){
        auto v = new vertex();
        v->addInputVertex(args...);
        return *v;
    }
    template <typename... ARGS>
    bool addReturn(ARGS&...args){
        exit.addInputVertex(args...);
    }
    vertex & getReturnVertex(){return exit;}
    vertex & getEntryVertex(){return entry;}
    private:
    vertex entry, exit;
};
}

#endif