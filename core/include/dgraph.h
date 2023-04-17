
#ifndef DGRAPH_H
#define DGRAPH_H
#include <vector>
#include <iostream>
#include <queue>
#include "exception.h"

namespace dgl
{

    class vertex;
    // a directional edge which can connect to multi vertices as output but only allow one vertex as input.
    class dedge
    {
    public:
        dedge() = default;
        dedge( const dedge& e){
            vertexFrom = e.getVertexFrom();
            auto vec = e.getVerticesTo();
            verticeTo.assign(vec.begin(), vec.end());
        }
        virtual ~dedge() {}
        void reset(){
            vertexFrom = nullptr; 
            verticeTo.clear();
        }
        void resetVerticesTo(){verticeTo.clear();}
        void resetVertexFrom(){vertexFrom = nullptr;}
        void attachTo(vertex *v) { verticeTo.push_back(v); }
        void attachFrom(vertex *v) { vertexFrom = v; }
        void connect(vertex* v1, vertex *v2);

        std::vector<vertex*> getVerticesTo() const { return verticeTo; }
        vertex * getVertexFrom() const { return vertexFrom; }

        vertex *vertexFrom = nullptr;
        std::vector<vertex*> verticeTo;
    };

    class vertex
    {
    public:
        vertex() = default;
        virtual ~vertex() {}
        vertex(const vertex &rhs)
        {
            bExplored = rhs.bExplored;

            inEdges.reserve(rhs.inEdges.size());
            copy(inEdges.begin(), inEdges.end(), inEdges.begin());
            outEdges.reserve(rhs.outEdges.size());
            copy(outEdges.begin(), outEdges.end(), outEdges.begin());
        }
        vertex(vertex &&rhs) : inEdges(std::move(rhs.getInEdges())),
                               outEdges(std::move(rhs.getOutEdges())),
                               bExplored(rhs.bExplored)
                               //_value(rhs._value)
        {
            rhs.~vertex();
        }
        //
        // attach this vertex to/from some dedges
        //
        template <typename... ARGS>
        void attachTo(ARGS* ...args)
        {
            auto dedges = {args...};
            for (auto& e : dedges)
            {
                auto ptr = dynamic_cast<dedge*>(e);
                outEdges.push_back(ptr);
                ptr->attachFrom(this);
            }
        }

        template <typename... ARGS>
        void attachFrom(ARGS* ...args)
        {
            auto dedges = {args...};
            for (auto& e : dedges)
            {
                inEdges.push_back(e);
                e->attachTo(this);
            }
        }

        // detach an input/output dedge from this vertex
        // detached the dedge e from this vertex w
        void detachOutput(dedge *e);
        void detachInput(dedge *e);

        // detach this vertex from its connected vertices
        void detach();

        bool isExplored(){ return bExplored; }
        void setExplored(bool key){ bExplored = key;}
        // detach this vertex from the graph. The in/out dedges will be disconnected
        // from this vertex. But these dedges won't be deleted as they are not part
        // of this vertex. 
        const std::vector<dedge *>& getInEdges() const { return inEdges; }
        const std::vector<dedge *>& getOutEdges() const { return outEdges; }
        void clearInEdges(){ inEdges.clear(); }
        void clearOutEdges(){ outEdges.clear(); }

        void reset() { bExplored = 0; }

        std::vector<dedge *> inEdges;
        std::vector<dedge *> outEdges;

        // check if all the outgoing vertice have been marked as explored
        //bool isExploredOugoingVertex();
        // check if all the incoming vertice have been marked as explored
        //bool isExploredIncomingVertex();
        bool setActivity(bool val){bActive = val;}
        bool isActive(){return bActive;}

        // status variable used by walk algorithm
        bool bExplored = 0;
        bool bActive = 1;
        size_t hashId =0;
        //int _value = 0;
        //void print() { std::cout << "vertex value: " << _value << std::endl; }
    };

    class graph {
    // A graph is a collection of vertices connected by dedges.
    // All the vertices contained in a graph is called subvertices.
    public :  
        graph() = default;
        virtual ~graph(){}
        //getVertices return the first level subvertices (entries)
        void addL1Vertex(vertex* vtx){
            l1Vertices.push_back(vtx);
        }
        template <typename callable>
        void BFWalk(callable &&fn){
            //The breadth-first walk through the graph starting at this vertex.
            //call the callable at the begining of visiting each vertex.
            //the callable should return void.
            //the fn will be executed on each vertex at most once.
            std::vector<vertex *> vertice_buffer;
            std::queue<vertex *> _vq;
            for(auto vtx : l1Vertices)
                _vq.push(vtx);
            while (_vq.size())
                {
                    auto v = _vq.front();
                    _vq.pop();
                    if(!(v->isActive())) continue;
                    vertice_buffer.push_back(v);
                    auto dedges = v->getOutEdges();
                    for (auto& e : dedges)
                    {
                        for(auto & vn : e->getVerticesTo())
                        {
                            if (!(vn->bExplored))
                            {
                                vn->bExplored = 1;
                                _vq.push(vn);
                            }
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
        std::vector<vertex* > l1Vertices;
    };

    // class tree
    // {
    // public:
    //     tree() = default;

    //     template <typename callable>
    //     void BFWalk(callable &fn);

    //     template <typename callable>
    //     void DFWalk(callable &fn);

    //     vertex *entry_vertex;
    // };
}


#endif