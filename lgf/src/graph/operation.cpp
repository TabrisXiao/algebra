#include "graph/object.h"
#include "graph/operation.h"
#include "graph/context.h"

lgf::sid_t lgf::operation::internal_rep()
{
    sid_t res = "";
    sid_t outrep = "";
    for (auto &e : outgoingEdges)
    {
        auto obj = dynamic_cast<object *>(e->endn);
        if (obj)
        {
            outrep += obj->represent() + ", ";
        }
    }
    if (outrep.size() > 0)
    {
        res += outrep.substr(0, outrep.size() - 2) + " = ";
    }
    res += name;
    sid_t attr_sid = "";
    for (auto &it : attributes)
    {
        attr_sid += it.first + ": " + it.second.represent() + ", ";
    }
    if (attr_sid.size() != 0)
    {
        res += " {" + attr_sid.substr(0, attr_sid.size() - 2) + "}";
    }

    sid_t inrep = "";
    for (auto &e : incomingEdges)
    {
        auto obj = dynamic_cast<object *>(e->startn);
        if (obj)
        {
            inrep += obj->get_sid() + ", ";
        }
    }
    if (inrep.size() > 0)
    {
        res += " (" + inrep.substr(0, inrep.size() - 2) + ")";
    }

    for (auto &r : regions)
    {
        res += r->represent();
    }
    return res;
}

lgf::object *lgf::operation::create_object(const description &desc)
{
    if (reg_ == nullptr)
    {
        throw std::runtime_error("The operation doesn't belong to any region.");
    }
    auto obj = reg_->add_node<object>(std::make_unique<object>(desc));
    this->link_to(obj);
    return obj;
}

lgf::context *lgf::operation::get_context()
{
    auto g = get_define_region();
    while (g)
    {
        context *c = dynamic_cast<context *>(g->get_defining_op());
        if (c != nullptr)
        {
            return c;
        }
        auto defop = g->get_defining_op();
        if (defop == nullptr)
        {
            return nullptr;
        }
        g = defop->get_define_region();
    }
    return nullptr;
}