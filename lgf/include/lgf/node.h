
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

    typedef size_t id_t;
    class node : public lgfObject
    {
    public:
        // the first output value is dependency value used to store
        // the dependency inform that don't have value connections
        node(std::string id = "op", graph *g = nullptr) : lgfObject(id)
        {
            _v_ = std::make_unique<value>();
            graph_ = g;
        };
        virtual ~node() = default;
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = " << get_sid() << ": " << inputs_sid();
            return p.dump();
        }

        descriptor get_value_desc()
        {
            return _v_->get_desc();
        }

        template <typename T>
        T *get_value_desc_as()
        {
            return dynamic_cast<T *>(_v_->get_desc());
        }

        value &get_value() { return *_v_; }

        void set_value_desc(descriptor d)
        {
            if (d.is_null())
            {
                throw std::runtime_error("set_value_desc: descriptor is null");
            }
            _v_->set_desc(d);
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
            auto it = std::find_if(inputs.begin(), inputs.end(), [n](edge &e)
                                   { return e.get_dual_node() == n; });
            if (it == inputs.end())
                return;
            inputs.erase(it);
        }

        void add_input_edge(edge &e)
        {
            inputs.push_back(std::move(e));
        }

        void add_output_edge(edge &&e)
        {
            users.push_back(std::move(e));
        }

        void link_to(edge &e)
        {
            users.push_back(edge(this));
            users.back().couple(e);
        }

        edgeBundle &get_user_handles()
        {
            users.clean();
            return users;
        }

        edgeBundle &get_input_handles()
        {
            inputs.clean();
            return inputs;
        }

        bool is_valid_handle(edge &e)
        {
            return e.is_coupled();
        }

        void add_input(node *n)
        {
            if (n == this)
                return;
            edge se(this);
            edge de(n);
            se.couple(de);
            inputs.push_back(std::move(se));
            n->add_output_edge(std::move(de));
        }

        template <typename... ARGS>
        void register_input(ARGS... args)
        {
            auto nodes = std::initializer_list<node *>{args...};
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

        void register_inputs_at(std::vector<node *> &inserts, size_t idx)
        {
            if (idx > inputs.size())
                throw std::runtime_error("register_inputs_at index out of input range");
            std::vector<edge> new_inputs;
            new_inputs.reserve(inputs.size() + inserts.size() + 1);
            auto size = inputs.size();
            for (size_t j = 0; j < idx; j++)
            {
                new_inputs.push_back(std::move(inputs[j]));
            }
            for (size_t j = 0; j < inserts.size(); j++)
            {
                if (inserts[j] == this)
                    return;
                edge se(this);
                edge de(inserts[j]);
                se.couple(de);
                inserts[j]->add_output_edge(std::move(de));
                new_inputs.push_back(std::move(se));
            }
            for (auto j = idx; j < size; j++)
            {
                new_inputs.push_back(std::move(inputs[j]));
            }

            inputs.swap(new_inputs);
        }

        std::vector<node *> get_users()
        {
            std::vector<node *> res;
            res.reserve(users.size());
            for (auto &e : users)
            {
                if (!e.is_coupled())
                    continue;
                res.push_back(e.get_dual_node());
            }
            return res;
        }

        std::vector<node *> get_input_nodes()
        {
            std::vector<node *> res;
            res.reserve(inputs.size());
            for (auto &e : inputs)
            {
                if (!e.is_coupled())
                    continue;
                res.push_back(e.get_dual_node());
            }
            return res;
        }

        void drop_input(node *n)
        {
            auto it = std::find_if(inputs.begin(), inputs.end(), [n](edge &e)
                                   { if( !e.is_coupled() ) return false;
                                    return e.get_dual_node() == n; });
            if (it == inputs.end())
                return;
            (*it).decouple();
            inputs.erase(it);
        }

        void replace_input_by(edge &h, node *n)
        {
            edge e(n);
            e.couple(h);
            n->add_output_edge(std::move(e));
        }

        void replace_input_by(node *on, node *nn)
        {
            auto it = std::find_if(inputs.begin(), inputs.end(), [on](edge &e)
                                   { if( !e.is_coupled() ) return false;
                                    return e.get_dual_node() == on; });
            if (it == inputs.end())
                return;
            (*it).decouple();
            nn->link_to(*it);
        }

        // this function assume that this op takes no inputs from the
        // new op. But allow this op is a input of the
        // new op.
        void replace_by(node *newop)
        {
            if (this == newop)
                return;
            for (auto &e : users)
            {

                if (e.is_coupled())
                {
                    // if the new node is a user of this node, skip it.
                    if (e.get_dual_node() == newop)
                        continue;
                    e.update_node(newop);
                    newop->add_output_edge(std::move(e));
                }
            }
        }

        node *input(size_t i = 0)
        {
            if (i >= inputs.size())
                throw std::runtime_error("input(size_t i): calling input index out of range");

            if (inputs[i].is_coupled())
                return inputs[i].get_dual_node();

            // if the edge is not coupled, it means that this input is
            // no longer valid and need to be removed.
            inputs.erase(inputs.begin() + i);
            return input(i);
        }

        node *user(size_t i)
        {
            if (i >= users.size())
                throw std::runtime_error("calling user index out of range");
            if (users[i].is_coupled())
                return users[i].get_dual_node();
            // if the edge is not coupled, it means that this user is
            // no longer valid and need to be removed.
            users.erase(users.begin() + i);
            return user(i);
        }

        template <typename T>
        T *get_user(size_t i)
        {
            return dynamic_cast<T *>(user(i));
        }

        value &input_value(size_t i)
        {
            return input(i)->get_value();
        }

        value &output()
        {
            return get_value();
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

        void assign_value_id(int &n);

        void set_exploration(bool a) { bExplored = a; }

        bool is_explored() { return bExplored; }

        bool is_deprecate() { return bDeprecate; }

        void deprecate()
        {
            bDeprecate = 1;
        }

        virtual void print();

        size_t get_input_size();
        size_t get_user_size() { return users.size(); }

        sid_t represent_inputs()
        {
            sid_t p;
            for (auto &e : inputs)
            {
                if (!e.is_coupled())
                    continue;
                p += e.get_dual_node()->get_value_sid() + ", ";
            }
            p.pop_back();
            p.pop_back();
            return p;
        }

        bool is_commutable() { return bCommutable; }
        void set_commutable(bool a) { bCommutable = a; }

        // this function only valid for tree structure graph.
        // std::string get_uid()
        // {
        //     std::string id = get_sid();
        //     id += '(';
        //     std::vector<std::string> vec;
        //     for (auto i = 0; i < inputs.size(); i++)
        //     {
        //         auto n = inputs[i].get_dual_node();
        //         vec.push_back(n->get_uid());
        //     }
        //     if (bCommutable)
        //     {
        //         std::sort(vec.begin(), vec.end());
        //     }
        //     for (auto i = 0; i < vec.size(); i++)
        //     {
        //         id = id + vec[i] + ',';
        //     }
        //     id.pop_back();
        //     id += ')';
        //     return id;
        // }

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

        graph *get_parent_graph() { return graph_; }

        void set_parent_graph(graph *g) { graph_ = g; }

        virtual std::string inputs_sid();

        bool is_dependency_fullfilled()
        {
            inputs.clean();
            for (auto &e : inputs)
            {
                if (!e.is_coupled())
                    continue;
                auto n = e.get_dual_node();
                if (n->is_explored() || n->is_deprecate())
                    continue;
                return false;
            }
            return true;
        }

        void reset_walk_status()
        {
            bExplored = 0;
        }

        void mark_status(size_t s)
        {
            status.shift(s);
        }
        void remove_status(size_t s)
        {
            status.clear(s);
        }
        bool get_status(size_t s)
        {
            return status.bit_check(s);
        }

    private:
        std::unique_ptr<value> _v_;
        // this function is used to determine if this node contained
        // a region. If an op contained a region, it should override
        // this function.
        // virtual graph* getSubgraph(){ return nullptr;}
        bool bExplored = 0;
        bool bCommutable = 0;
        // this is a member used to remove the node efficiently.
        // Should be used solely for removing process in graph.
        bool bDeprecate = 0;
        // this status is added for customized usage.
        bitCode<size_t> status;
        edgeBundle inputs;
        edgeBundle users;
        graph *graph_ = nullptr;
    };

    class graph : public node
    {
    public:
        graph() = default;

        graph(std::string id, graph *pg = nullptr)
            : node(id, pg) {}

        virtual void print() override;

        virtual std::string represent() override { return ""; }

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
                if (node->is_deprecate() || node->is_explored())
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
                    // need to skip invalid edges
                    if (!(h.is_coupled()))
                    {
                        continue;
                    }
                    auto vn = h.get_dual_node();
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

        virtual void print_graph(int &id_start);

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