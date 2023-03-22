
#ifndef SDGRAPH_H
#define SDGRAPH_H
#include <vector>
#include <iostream>
#include <queue>
// Simplified Directed Graph Library
// Only vertex is used to construct graphs. The edge is represented by connection
// between vertices.
namespace sdgl{

class vertex;
class vertex{
public : 
    vertex () =default;
    virtual ~vertex() {}
    vertex(vertex &rhs)
    {
        bExplored = rhs.bExplored;
    }
    vertex(vertex &&rhs) : outVertices(std::move(rhs.getOutVertices())),
                           inVertices(std::move(rhs.getInVertices())),
                           bExplored(rhs.bExplored)
    {
        rhs.~vertex();
    }
    // std::vector<vertex*> & getInput(){return inVertices;}
    // std::vector<vertex*> & getOutput(){return outVertices;}
    //
    // attach this vertex to/from other vertices
    //
    void addOutgoingVertex(vertex *v){ outVertices.push_back(v); }
    void addIncomingVertex(vertex *v){ inVertices.push_back(v); }

    bool isLinkedTo(vertex *v){
        for(auto _v : outVertices){
            if(v==_v) return true;
        }
        return false;
    }
    bool isLinkedFrom(vertex *v){
        for(auto _v : inVertices){
            if(v==_v) return true;
        }
        return false;
    }
    // linkFrom 
    template <typename... ARGS>
    void linkFrom(ARGS &...args)
    {
        auto vertices = {&args...};
        for (auto v : vertices)
        {
            auto ptr = dynamic_cast<vertex*>(v);
            if(isLinkedFrom(ptr)) continue;
            inVertices.push_back(ptr);
            ptr->addOutgoingVertex(this);
        }
    }
    template <typename... ARGS>
    void linkTo(ARGS &...args)
    {
        auto vertices = {&args...};
        for (auto v : vertices)
        {
            auto ptr = dynamic_cast<vertex*>(v);
            if(isLinkedTo(ptr)) continue;
            outVertices.push_back(ptr);
            ptr->addIncomingVertex(this);
        }
    }
    // detach the vertex from this vertex if this vertex linked to it.
    void detachOutgoingVertex(vertex *v_);
    // detach the vertex from this vertex if this vertex linked from it.
    void detachIncomingVertex(vertex *v_);
    // detach this vertex from any other vertices linked to it.
    void detach();
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
        this->bExplored=1;
        _vq.push(this);
        while (_vq.size())
            {
                auto v = _vq.front();
                if(!bActive) continue;
                _vq.pop();
                vertice_buffer.push_back(v);
                auto vertices = v->getOutVertices();
                for (auto &vn : vertices)
                {
                    if (!(vn->bExplored))
                    {
                        vn->bExplored= 1;
                        _vq.push(vn);
                    }
                }
                fn(v);
            }
            for (auto &v : vertice_buffer)
            {
                v->reset();
            }
        return;
    }
    std::vector<vertex*> & getInVertices(){return inVertices;}
    std::vector<vertex*> & getOutVertices(){return outVertices;}
    void setActivation( bool b){bActive = b;}
    virtual bool isActive(){return bActive;}

    bool bExplored = 0;
    bool bActive = 1;
    std::vector<vertex*> outVertices;
    std::vector<vertex*> inVertices;
};

class chain {
    public : 
    chain () = default;
    vertex head, tail;
    int size_=0;
};

class treeNode : public vertex {
public:
    treeNode() = default;
    std::vector<treeNode> children;
};

class sdgraph {
    public : 
    sdgraph () = default;
    virtual ~sdgraph(){
        entry.BFWalk([](vertex* vtx){
            delete vtx;
        });
    }
    void addL1Vertex(vertex* vtx){
        entry.linkTo(*vtx);
    }
    vertex & getEntry(){return entry;}
    vertex & getReturn(){return exit;}
    template <typename... ARGS>
    bool addReturn(ARGS&...args){
        exit.addIncomingVertex(args...);
    }
    vertex & getReturnVertex(){return exit;}
    vertex & getEntryVertex(){return entry;}
    private:
    vertex entry, exit;
};
}

#endif