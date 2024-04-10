
#ifndef node_H_
#define node_H_
#include "object.h"
#include "edge.h"
#include "value.h"
#include "context.h"
#include <unordered_set>
#include <queue>
#include <map>
#include <set>
#include <string>
#include "printer.h"
#include "global.h"
#include "exception.h"
#include <memory>
#include <algorithm>
#include "utils.h"

// logic graph frameworks
namespace lgf
{
    class graph;
    class normalizer;

    typedef size_t id_t;
    typedef std::unique_ptr<edge> edgeHandle;
    class node : public graphObject
    {
    public:
        // the first output value is dependency value used to store
        // the dependency inform that don't have value connections
        node(std::string id = "op", graph *g = nullptr) : graphObject(id)
        {
            _v_ = std::make_unique<value>();
            graph_ = g;
        };
        virtual ~node() = default;
        virtual std::string represent()
        {
            printer p;
            p << _v_->get_sid() << " = " << get_sid() << " : " << inputs_sid();
            return p.dump();
        }

        valueDesc *get_value_desc() { return _v_->get_desc(); }

        value &get_value() { return *_v_; }

        void set_value_desc(valueDesc *desc)
        {
            _v_->set_desc(desc);
        }

        sid_t get_value_sid()
        {
            return _v_->get_sid();
        }

        sid_t value_rep()
        {
            return _v_->represent();
        }

        sid_t value_desc_rep()
        {
            return _v_->desc_represent();
        }

        void erase_input(node *n)
        {
            auto it = std::find_if(inputs.begin(), inputs.end(), [n](edgeHandle &e)
                                   { if(!e) return false;
                                    return e->get_dual_node() == n; });
            if (it == inputs.end())
                return;
            inputs.erase(it);
        }

        void add_input_edge(edgeHandle &e)
        {
            inputs.push_back(std::move(e));
        }

        void add_output_edge(edgeHandle &e)
        {
            users.push_back(std::move(e));
        }

        void link_to(edgeHandle &e)
        {
            users.push_back(std::make_unique<edge>(this));
            users.back()->couple(e.get());
        }

        std::vector<edgeHandle> &get_user_handles()
        {
            return users;
        }

        std::vector<edgeHandle> &get_input_handles()
        {
            return inputs;
        }

        void add_input(node *n)
        {
            if (n == this)
                return;
            edgeHandle se = std::make_unique<edge>(this);
            edgeHandle de = std::make_unique<edge>(n);
            se->couple(de.get());
            inputs.push_back(std::move(se));
            n->add_output_edge(std::move(de));
        }

        template <typename... ARGS>
        void register_input(ARGS... args)
        {
            auto nodes = std::initializer_list<node*>{args...};
            for (auto n : nodes)
            {
                add_input(n);
            }
        }

        void register_inputs(std::vector<node *> &ns)
        {
            for (auto n : ns)
            {
                register_input(n);
            }
        }

        void replace_input_by(node *on, node *nn)
        {
            auto it = std::find_if(inputs.begin(), inputs.end(), [on](edgeHandle &e)
                                   { if(!e) return false;
                                    return e->get_dual_node() == on; });
            if (it == inputs.end())
                return;
            (*it)->decouple();
            nn->link_to(*it);
        }

        // this function regardless if the case that if there was any
        // mutual links between this node and the new node.
        // so it may cause the cycle dependence.
        void replace_by(node *n)
        {
            if (this == n)
                return;
            for (auto &e : users)
            {
                if(e && e->is_coupled())
                {
                    e->update_node(n);
                }
            }
            users.swap(n->users);
        }

        node *input(size_t i=0)
        {
            if (i >= inputs.size())
                throw std::runtime_error("calling input index out of range");

            if (inputs[i]->is_coupled())
                return inputs[i]->get_dual_node();

            // if the edge is not coupled, it means that this input is
            // no longer valid and need to be removed.
            inputs.erase(inputs.begin() + i);
            return input(i);
        }

        node *user(size_t i)
        {
            if (i >= users.size())
                throw std::runtime_error("calling user index out of range");
            if (users[i]->is_coupled())
                return users[i]->get_dual_node();
            // if the edge is not coupled, it means that this user is
            // no longer valid and need to be removed.
            users.erase(users.begin() + i);
            return user(i);
        }

        value &input_value(size_t i)
        {
            return input(i)->get_value();
        }

        value &output()
        {
            return get_value();
        }

        void infer_trivial_value_desc()
        {
            if (!inputs.size())
                return;
            get_value().set_desc(input_value(0).get_desc());
        }

        // drop all inputs to this node, and remove all connects
        // associated to the op.
        void drop_all_inputs()
        {
            inputs.clear();
        }

        void erase()
        {
            inputs.clear();
            users.clear();
            deprecate();
        }

        void set_nontrivial() { bTrivial = 0; }

        bool is_trivial() { return bTrivial; }

        void assign_value_id(int &n);

        void set_exploration(bool a) { bExplored = a; }

        bool is_explored() { return bExplored; }

        bool is_deprecate() { return bDeprecate; }

        void deprecate()
        {
            bDeprecate = 1;
        }

        virtual void print();

        size_t get_input_size() const;
        size_t get_user_size() const { return users.size(); }

        std::string get_op_represent()
        {
            // get representation of this node after the first "="
            auto code = represent();
            auto pos = code.find("=");
            if (pos != std::string::npos)
                return code.substr(pos + 1);
            else
                return code;
        }

        bool is_identical(node *target)
        {
            if (this == target)
                return true;
            if (target == nullptr)
                return false;
            auto code1 = this->get_op_represent();
            auto code2 = target->get_op_represent();
            if (code1 != code2)
                return false;
            return true;
        }

        graph *get_parent_graph() { return graph_; }

        void set_parent_graph(graph *g) { graph_ = g; }

        virtual std::string inputs_sid();

        bool is_dependency_fullfilled()
        {
            for (auto &e : inputs)
            {
                if (!e->is_coupled())
                    continue;
                if (e->get_dual_node()->is_explored())
                    continue;
                return false;
            }
            return true;
        }

        void reset_walk_status()
        {
            bExplored = 0;
        }

    private:
        std::unique_ptr<value> _v_;
        // this function is used to determine if this node contained
        // a region. If an op contained a region, it should override
        // this function.
        // virtual graph* getSubgraph(){ return nullptr;}
        bool bExplored = 0;
        bool bTrivial = 1;

        // this is a member used to remove the node efficiently.
        // Should be used solely for removing process in graph.
        bool bDeprecate = 0;
        std::vector<std::unique_ptr<edge>> inputs;
        std::vector<std::unique_ptr<edge>> users;
        graph *graph_ = nullptr;
    };

    class graph : public node
    {
    public:
        graph() = default;

        graph(std::string id, graph *pg = nullptr)
            : node(id, pg) {}

        virtual void print() override;

        virtual std::string represent() { return ""; }

        // return how many nodes graph contained
        size_t get_node_size() { return nodes.size(); }

        LGFContext &get_context() { return ctx; }

        // note that this function will not take care of the inputs and users
        // replacement. It only replace the node in the nodes container.
        void replace_node(node *old, node *new_op);

        // A breadth-first walk function that is graph modification safe.
        // Call the callable at the begining of visiting each vertex.
        // The callable should return void.
        // The fn will be executed on each node at most once.
        // The node ran by fn is marked as done. A node will
        // got processed only if all inputs nodes are done (All
        // dependences are processed).
        // notice that this walk skipped the entryOp so that we don't
        // need to worry about the entry op got modified by accident.
        template <typename callable>
        void walk(callable &&fn, bool deepWalk = 0)
        {
            std::queue<node *> _vq;
            std::vector<node *> vertice_buffer;
            vertice_buffer.reserve(get_node_size());
            for (auto node : nodes)
            {
                if (node->get_input_size() != 0)
                    continue;
                _vq.push(node);
            }
            while (_vq.size())
            {
                auto v = _vq.front();
                _vq.pop();
                if (v->is_deprecate() || v->is_explored())
                    continue;
                v->set_exploration(true);

                vertice_buffer.push_back(v);
                for (auto &h : v->get_user_handles())
                {
                    // need to skip invalid edgeHandles
                    if(!h || !(h->is_coupled()))
                    {
                        continue;
                    }
                    auto vn = h->get_dual_node();
                    if (vn->is_explored() || vn->is_deprecate() || !(vn->is_dependency_fullfilled()))
                        continue;
                    _vq.push(vn);
                }

                if (deepWalk)
                {
                    if (auto g = dynamic_cast<graph *>(v))
                    {
                        g->walk(fn, 1);
                    }
                }
                fn(v);
            }
            for (auto v : vertice_buffer)
            {
                v->reset_walk_status();
            }
            clean();
            return;
        }

        graph *get_graph() { return dynamic_cast<graph *>(this); }

        virtual void print_graph(int & id_start );

        void assign_id(int n = 0);
        std::vector<node *>::iterator insert_after(node *op, node *new_op)
        {
            auto iter = std::find(nodes.begin(), nodes.end(), op);
            if (iter == nodes.end())
            {
                nodes.push_back(new_op);
                return nodes.end();
            }
            else
            {
                return nodes.insert(iter + 1, new_op);
            }
        }

        // clean will remove all nodes marked as is_deprecate;
        // return 0 if no ops got removed. Otherwise return 1;
        bool clean();

        std::vector<node *> &get_nodes() { return nodes; }

    private:
        std::vector<node *> nodes;
        LGFContext ctx;
        // how many nodes contained in this graph
    };

}

#endif