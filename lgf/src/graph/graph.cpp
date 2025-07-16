#include "graph/graph.h"

void lgfc::node::detach()
{
    for (auto &e : incomingEdges)
    {
        e->outgoing()->remove_outgoing_edge(e);
    }
    for (auto &e : outgoingEdges)
    {
        e->incoming()->remove_incoming_edge(e);
    }
    incomingEdges.clear();
    outgoingEdges.clear();
}

void lgfc::node::link_to(node *n)
{
    if (n == nullptr)
        return;
    auto e = std::make_shared<edge>(this, n);
    n->add_input_edge(e);
    outgoingEdges.push_back(std::move(e));
}