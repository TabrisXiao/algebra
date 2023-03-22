
#ifndef DGRAPH_H
#define DGRAPH_H
#include <vector>
#include <iostream>
#include <queue>

namespace dgl
{

    class vertex;
    class edge
    {
    public:
        edge() = default;
        virtual ~edge() {}
        void reset(){
            vertexFrom = nullptr; 
            vertexTo = nullptr;
        }
        void resetReceiver(){vertexTo = nullptr;}
        void resetSender(){vertexFrom = nullptr;}
        void attachTo(vertex &v) { vertexTo = &v; }
        void attachFrom(vertex &v) { vertexFrom = &v; }
        void connect(vertex &vfrom, vertex &vto);

        vertex *getReceiver() { return vertexTo; }
        vertex *getSender() { return vertexFrom; }

        vertex *vertexFrom = nullptr, *vertexTo = nullptr;
    };
    class vertex
    {
    public:
        vertex() = default;
        virtual ~vertex() {}
        vertex(vertex &rhs)
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
        // attach this vertex to/from some edges
        //
        template <typename... ARGS>
        void attachTo(ARGS &...args)
        {
            auto edges = {&args...};
            for (auto e : edges)
            {
                auto ptr = dynamic_cast<edge*>(e);
                outEdges.push_back(ptr);
                ptr->attachFrom(*this);
            }
        }

        template <typename... ARGS>
        void attachFrom(ARGS &...args)
        {
            auto edges = {&args...};
            for (auto e : edges)
            {
                inEdges.push_back(e);
                e->attachTo(*this);
            }
        }

        // detach an input/output edge from this vertex
        // detached the edge e from this vertex w
        void detachOutput(edge *e);
        void detachInput(edge *e);

        // detach this vertex from its connected vertices
        void detach();
        template <typename callable>
        vertex *walk(vertex *entry_v, callable &fn)
        {
            // the callable function should return a bool to indicate
            // if the walk should be stopped or not (true for stop)
            std::vector<vertex *> vertice_buffer;

            vertex *return_ptr = nullptr;
            std::queue<vertex *> _vq;
            _vq.push(entry_v);

            while (_vq.size())
            {
                auto v = _vq.front();
                _vq.pop();
                auto bkey = fn(v);
                v->bExplored = 1;
                if (bkey)
                {
                    return_ptr = v;
                    break;
                }
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
            return return_ptr;
        }

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
                    if(!(v->isActive())) continue;
                    if(v->bExplored) continue;
                    vertice_buffer.push_back(v);
                    auto edges = v->getOutEdges();
                    fn(v);
                    v->setExplored(1);
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

        bool isExplored(){ return bExplored; }
        void setExplored(bool key){ bExplored = key;}
        // detach this vertex from the graph. The in/out edges will be disconnected
        // from this vertex. But these edges won't be deleted as they are not part
        // of this vertex. 
        std::vector<edge *> &getInEdges() { return inEdges; }
        std::vector<edge *> &getOutEdges() { return outEdges; }

        void reset() { bExplored = 0; }

        std::vector<edge *> inEdges;
        std::vector<edge *> outEdges;

        // check if all the outgoing vertice have been marked as explored
        bool isExploredOugoingVertex();
        // check if all the incoming vertice have been marked as explored
        bool isExploredIncomingVertex();
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
    // A graph is a collection of vertices connected by edges.
    // All the vertices contained in a graph is called subvertices.
    public :  
        graph() = default;
        virtual ~graph(){}
        //getVertices return the first level subvertices (entries)
        void addL1Vertex(vertex* vtx){
            l1Vertices.push_back(vtx);
        }
        void replaceVertex(vertex * origVtx, vertex* newVtx){
            for(auto e : origVtx->getInEdges()){
                newVtx->attachFrom(*e);
            }
            origVtx->getInEdges().clear();
            for(auto e : origVtx->getOutEdges()){
                newVtx->attachTo(*e);
            }
            origVtx->getOutEdges().clear();
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
                    if(!(v->isActive()) || v->isExplored()) continue;
                    vertice_buffer.push_back(v);
                    auto edges = v->getOutEdges();
                    fn(v);
                    v->bExplored = 1;
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