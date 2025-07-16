#include "graph/object.h"
#include "graph/operation.h"

lgf::operation *lgf::object::get_defining_op()
{
    operation *res = nullptr;
    if (incomingEdges.size() > 0)
    {
        res = dynamic_cast<operation *>(incomingEdges[0]->startn);
    }
    return res;
}

std::vector<lgf::operation *> lgf::object::get_users()
{
    std::vector<operation *> res;
    for (auto &e : outgoingEdges)
    {
        operation *op = dynamic_cast<operation *>(e->endn);
        if (op)
        {
            res.push_back(op);
        }
    }
    return res;
}