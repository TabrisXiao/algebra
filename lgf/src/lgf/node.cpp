
#include <unordered_set>
#include <algorithm>
#include "lgf/node.h"
using namespace lgf;

void value::print() { global::stream::getInstance() << represent() << "\n"; };
//---------------------------------------------------

//////////////////////////////////////////////////////

std::string node::default_inputs_represent()
{
    if (get_input_size() == 0)
        return "";
    printer p;
    p << inputs[0].get_link_node()->get_value().get_sid();
    for (auto iter = inputs.begin() + 1; iter != inputs.end(); iter++)
    {
        p << ", " << (*iter).get_link_node()->get_value().get_sid();
    }
    return p.dump();
}

//---------------------------------------------------
void node::print()
{
    global::stream::getInstance().printIndent();
    global::stream::getInstance() << represent() << "\n";
}

//---------------------------------------------------
void node::assign_value_id(int &n)
{
    _v_.get()->set_sid("%" + std::to_string(n));
}

//---------------------------------------------------
size_t node::get_input_size()
{
    return inputs.size();
}

void node::create_region(int n)
{
    for (auto i = 0; i < n; i++)
    {
        regions.emplace_back(region(this));
    }
}

bool node::is_isomorphic(node *n)
{
    if (n == this)
        return true;
    if (n->get_input_size() != get_input_size() || get_dict().size() != n->get_dict().size())
        return false;

    if (inputs.size() != 0)
    {
        // check attributes

        for (auto &attr : get_dict())
        {
            auto pattr = n->get_attr(attr.first);
            if (pattr.is_null())
                return false;
            if (attr.second.represent() != pattr.represent())
                return false;
        }
        // check if the inputs are isomorphic

        for (auto i = 0; i < inputs.size(); i++)
        {
            auto node = inputs[i].get_link_node();
            auto dnode = n->inputs[i].get_link_node();
            // in case hit a loop.
            if (node == n || dnode == this)
                return false;
            if (!node->is_isomorphic(dnode))
                return false;
        }
    }
    else
    {
        if (represent() != n->represent())
            return false;
    }
    return true;
}

//////////////////////////////////////////////////////

void region::replace_node(node *old, node *new_op)
{
    auto iter = std::find(nodes.begin(), nodes.end(), old);
    if (iter == nodes.end())
        return;
    *iter = new_op;
}

void region::print()
{
    global::stream::getInstance().printIndent();
    std::string code = "";
    // add space if the represent is not empty
    // {} no reprsent, shoudn't have space
    // module {}, have represent "module", should have space
    // between "module" and the {}.
    if (!code.empty())
        code += " ";
    global::stream::getInstance() << code;
    int id = 0;
    if (nodes.size() != 0)
        print_region(id);
    global::stream::getInstance() << "\n";
}
//---------------------------------------------------

void region::print_region(int &id_start)
{
    global::stream::getInstance() << "{\n";
    global::stream::getInstance().incrIndentLevel();

    walk([&id_start](node *op)
         {
        op->assign_value_id(id_start);
        id_start++;
        op->print(); },
         1);
    global::stream::getInstance().decrIndentLevel();
    global::stream::getInstance().printIndent();
    global::stream::getInstance() << "}";
}

void region::assign_id(int n0)
{
    int n = n0;
    walk([&n](node *op)
         {
        op->assign_value_id(n);
        n++; },
         1);
}
//---------------------------------------------------

bool region::clean()
{
    bool check = 0;
    for (auto iter = nodes.begin(); iter != nodes.end();)
    {
        node *op = (*iter);
        if (op->is_deprecate())
        {
            iter = nodes.erase(iter);
            check = 1;
            delete op;
        }
        else if (op->get_region_size() != 0)
        {
            for (auto &reg : op->get_regions())
            {
                check = reg.clean();
            }
            iter++;
        }
        else
            iter++;
    }
    return check;
}
//---------------------------------------------------