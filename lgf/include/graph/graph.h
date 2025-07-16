#ifndef LGF_GRAPH_H_
#define LGF_GRAPH_H_
#include <deque>
#include <algorithm>
#include <iostream>

namespace lgfc
{
    class edge;

    class node
    {
    public:
        node() = default;
        virtual ~node() = default;
        void link_to(node *n);
        void add_input_edge(const std::shared_ptr<edge> &e)
        {
            incomingEdges.push_back(e);
        }
        void add_output_edge(const std::shared_ptr<edge> &e)
        {
            outgoingEdges.push_back(e);
        }
        void remove_incoming_edge(std::shared_ptr<edge> &e)
        {
            auto iter = std::remove_if(incomingEdges.begin(), incomingEdges.end(),
                                       [&](const std::shared_ptr<edge> &p)
                                       {
                                           return p.get() == e.get();
                                       });
            incomingEdges.erase(iter, incomingEdges.end());
        }
        void remove_outgoing_edge(std::shared_ptr<edge> &e)
        {
            auto iter = std::remove_if(outgoingEdges.begin(), outgoingEdges.end(),
                                       [&](const std::shared_ptr<edge> &p)
                                       {
                                           return p.get() == e.get();
                                       });
            outgoingEdges.erase(iter, outgoingEdges.end());
        }

        void detach();

        std::deque<std::shared_ptr<edge>> incomingEdges, outgoingEdges;
    };
    class edge
    {
    public:
        edge() = default;
        edge(node *from, node *to)
        {
            startn = from;
            endn = to;
        }
        virtual ~edge() = default;
        node *incoming() const
        {
            return endn;
        }
        node *outgoing() const
        {
            return startn;
        }

        node *startn, *endn;
    };

    class graph
    {
    public:
        graph() = default;
        node *add_node(std::unique_ptr<node> n)
        {
            auto res = n.get();
            nodes.push_back(std::move(n));
            return res;
        }
        template <typename T>
        T *add_node(std::unique_ptr<node> &&n)
        {
            auto res = n.get();
            nodes.push_back(std::move(n));
            return dynamic_cast<T *>(res);
        }
        std::deque<std::unique_ptr<node>> nodes;
    };
}
#endif // GRAPH_GRAPH_H