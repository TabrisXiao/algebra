#ifndef LGF_OPERATION_H
#define LGF_OPERATION_H
#include "base.h"
#include "object.h"
#include "graph.h"
#include "attribute.h"
#include <map>
#include <queue>

namespace lgf
{
    using namespace lgfc;
    // a canvas has to have a static build function implemented to create an instance of the operation.
    class region;
    class context;
    class operation : public node, public base
    {
    public:
        operation(std::string name_, region *r = nullptr) : name(name_), reg_(r)
        {
        }
        std::string get_name()
        {
            return name;
        }
        object &output(size_t n)
        {
            if (n >= outgoingEdges.size())
            {
                throw std::out_of_range("Index out of range in operation::output");
            }
            return *dynamic_cast<object *>(outgoingEdges[n]->endn);
        }
        object &input(size_t n)
        {
            if (n >= incomingEdges.size())
            {
                throw std::out_of_range("Index out of range in operation::input");
            }
            return *dynamic_cast<object *>(incomingEdges[n]->startn);
        }
        size_t get_input_size() const
        {
            return incomingEdges.size();
        }
        size_t get_output_size() const
        {
            return outgoingEdges.size();
        }
        region *get_define_region()
        {
            return reg_;
        }

        template <typename... ARGS>
        void register_input(ARGS... args)
        {
            auto ins = std::initializer_list<node *>{args...};
            for (auto &n : ins)
            {
                n->link_to(this);
            }
        }

        void set_define_region(region *r)
        {
            reg_ = r;
        }

        template <typename T>
        std::unique_ptr<operation> clone()
        {
            auto new_op = std::make_unique<T>();
            for (auto &ie : incomingEdges)
            {
                auto obj = dynamic_cast<object *>(ie->startn);
                new_op->register_input(obj);
            }
            for (auto &oe : outgoingEdges)
            {
                auto obj = dynamic_cast<object *>(oe->endn);
                new_op->create_object(obj->get_desc());
            }
            new_op->attributes = attributes; // copy attributes
            return std::move(new_op);
        }

        virtual std::unique_ptr<operation> copy() = 0;

        object *create_object(const description &desc);

        context *get_context();
        void add_region()
        {
            regions.push_back(std::make_unique<region>(this));
        }
        region *get_region(size_t n = 0)
        {
            if (n >= regions.size())
            {
                throw std::out_of_range("Index out of range in operation::get_region");
            }
            return regions[n].get();
        }

        attribute get_attr(std::string name)
        {
            auto it = attributes.find(name);
            if (it != attributes.end())
            {
                return it->second;
            }
            else
            {
                throw std::runtime_error("Attribute not found");
            }
        }
        void set_attr(std::string name, attribute att)
        {
            attributes[name] = att;
        }

        // internal represent is a sid_t can be identify if two operations are equivalent or not.
        sid_t internal_rep();
        const bool is_valid() const
        {
            return isValid;
        }
        const bool is_visited() const
        {
            return isVisted;
        }
        void set_valid(bool v)
        {
            isValid = v;
        }
        void set_visited(bool v)
        {
            isVisted = v;
        }
        // update dependency counter, increase the dependency counter by 1.
        // return if the operation is ready to run.
        inline bool update_dependency()
        {
            dc_++;
            return (dc_ == get_input_size());
        }
        inline void reset_dependency()
        {
            dc_ = 0;
        }

    private:
        std::map<std::string, attribute> attributes;  // the attributes of the operation
        std::vector<std::unique_ptr<region>> regions; // the regions of the operation, it can be empty;
        // The following parameters are status flag for the operation.
        bool isValid = true;   // whether the operation is valid
        bool isVisted = false; // whether the operation has been visited in the graph traversal
        std::string name;
        size_t dc_ = 0;         // dependency counter, used to check if the operation is ready to run.
        region *reg_ = nullptr; // define region: the region that the operation belongs to, it can be null.
    };

    // a region is a set of operations. a region has to belong to an operation.
    class region : public graph
    {
    public:
        region() = delete;
        region(operation *p) : belong(p) {}
        operation *get_defining_op() const
        {
            return belong;
        }
        void set_defining_op(operation *p)
        {
            belong = p;
        }

        virtual std::string represent()
        {
            std::string res = "";
            for (auto &n : nodes)
            {
                auto op = dynamic_cast<operation *>(n.get());
                if (!op)
                    continue;
                res += op->represent() + "\n";
            }
            if (res.size())
            {
                res = "{\n" + res + "}\n";
            }
            else
            {
                res = "{}\n";
            }
            return res;
        }

        // clean is a function to remove ops that is no longer valid, is_valid() = false;
        //
        void clean()
        {
            auto iter = std::remove_if(nodes.begin(), nodes.end(),
                                       [](const std::unique_ptr<node> &n)
                                       {
                                           auto op = dynamic_cast<operation *>(n.get());
                                           if (!op)
                                           {
                                               return true;
                                           }
                                           else if (!op->is_valid())
                                           {
                                               op->detach();
                                               return true; // remove this node
                                           }
                                           return false; // keep this node
                                       });
            nodes.erase(iter, nodes.end());
        }

        void add_start_node(node *n)
        {
            start_nodes.push_back(n);
        }

        template <typename callable>
        void walk(callable &&fn)
        {
            std::queue<operation *> q;
            for (auto &n : start_nodes)
            {
                auto op = dynamic_cast<operation *>(n);
                if (op && op->is_valid())
                {
                    q.push(op);
                    continue;
                }
                auto ob = dynamic_cast<object *>(n);
                if (ob)
                {
                    auto users = ob->get_users();
                    for (auto &u : users)
                    {
                        auto op = dynamic_cast<operation *>(u);
                        if (op && op->is_valid())
                        {
                            q.push(op);
                        }
                    }
                }
            }
            for (auto &n : nodes)
            {
                auto op = dynamic_cast<operation *>(n.get());
                if (op && op->is_valid() && op->get_input_size() == 0)
                {
                    q.push(op);
                }
            }
            while (!q.empty())
            {
                auto op = q.front();
                q.pop();
                if (op->is_visited())
                    continue;
                op->set_visited(true);
                for (size_t i = 0; i < op->get_output_size(); i++)
                {
                    auto &out = op->output(i);
                    auto users = out.get_users();
                    for (auto &u : users)
                    {
                        auto next_op = dynamic_cast<operation *>(u);
                        if (next_op && !next_op->is_visited() && next_op->is_valid())
                        {
                            if (next_op->update_dependency())
                            {
                                q.push(next_op);
                            }
                        }
                    }
                }
                fn(op);
            }
            clean();
            for (auto &n : nodes)
            {
                auto op = dynamic_cast<operation *>(n.get());
                if (op)
                {
                    op->set_visited(false);
                    op->reset_dependency();
                }
            }
        }

        void assign_sid(int opid = 0, int obid = 0)
        {
            walk([&obid, &opid](operation *op)
                 {
                    if (op->get_sid().empty())
                    {
                        op->set_sid("op" + std::to_string(opid++));
                    }
                    for (auto &e : op->outgoingEdges)
                    {
                        auto obj = dynamic_cast<object *>(e->endn);
                        if (obj && obj->get_sid().empty())
                        {
                            obj->set_sid("%" + std::to_string(obid++));
                        }
                    } });
        }

        std::deque<node *> start_nodes;          // the start nodes of the region, it is used to traverse the region
        std::deque<std::unique_ptr<node>> nodes; // the nodes in the region
        operation *belong = nullptr;
        region *reg_ = nullptr; // define region
    };
}

// namespace lgf
#endif