
#ifndef AOG_H_
#define AOG_H_
#include <iostream>
#include <unordered_set>
#include <memory.h>
#include <algorithm>
#include "global.h"
#include "node.h"
// abstract node graph

namespace lgf
{

    class rewriterBase;

    class LGFContext;

    class painter
    {
    public:
        struct paintPoint
        {
            graph *g = nullptr;
            std::vector<node *> *nodes = nullptr;
            std::vector<node *>::iterator iter = std::vector<node *>::iterator();

            bool is_invalid()
            {
                if (g == nullptr || nodes == nullptr)
                    return true;
                if (iter < nodes->begin() || iter > nodes->end())
                    return true;
                return false;
            }
        };
        painter() = default;
        painter(graph *g) : point({g, &(g->get_nodes()), g->get_nodes().end()}), ctx(&(g->get_context())) {}
        painter(painter &p)
            : point(p.get_paintPoint()), ctx(p.get_context()) {}
        ~painter() {}
        void set_context(LGFContext *ctx_)
        {
            ctx = ctx_;
        }

        void set_paintPoint_to_graph_begin()
        {
            point.iter = point.g->get_nodes().begin();
        }
        void set_paintPoint_before(node *op)
        {
            point.g = op->get_parent_graph();
            auto &vec = point.g->get_nodes();
            point.iter = std::find(vec.begin(), vec.end(), op);
            if (point.iter != vec.begin())
                point.iter--;
        }
        void set_paintPoint_after(node *op)
        {
            point.g = op->get_parent_graph();
            auto &vec = point.g->get_nodes();
            point.iter = std::find(vec.begin(), vec.end(), op);
            if (point.iter != vec.end())
                point.iter++;
        }

        void reset_paintPoint()
        {
            if (!point.g)
                return;
            point.iter = point.g->get_nodes().end();
        }

        template <typename obj>
        obj *sketch()
        {
            auto op = obj::build();
            // op->inferType(ctx);
            return op;
        }

        template <typename obj, typename... ARGS>
        obj *sketch(ARGS... args)
        {
            auto op = obj::build(args...);
            // op->inferType(ctx);
            return op;
        }

        template <typename obj, typename... ARGS>
        obj *paint(ARGS... args)
        {
            THROW_WHEN(point.is_invalid(), "paint point is invalid!")
            // CHECK_CONDITION(point.g!=nullptr, "No graph associated to the painter!");
            auto op = sketch<obj>(args...);
            // add to graph
            op->set_parent_graph(point.g);
            point.iter = point.nodes->insert(point.iter, op) + 1;
            return op;
        }

        template <typename obj>
        obj *paint()
        {
            THROW_WHEN(point.is_invalid(), "paint point is invalid!")
            auto op = sketch<obj>();
            // add to graph
            op->set_parent_graph(point.g);
            point.iter = point.nodes->insert(point.iter, op) + 1;
            return op;
        }

        void insert_op(node *op)
        {
            point.iter = point.nodes->insert(point.iter, op) + 1;
        }

        template <typename obj>
        obj *replace_op(node *op1)
        {
            auto op2 = sketch<obj>();
            // There's no input for op1 so we can just replace
            // them directly
            op1->replace_by(op2);
            auto &nodes = point.g->get_nodes();
            auto iter = std::find(nodes.begin(), nodes.end(), op1);
            *iter = op2;
            op2->erase();
        }

        template <typename obj, typename... ARGS>
        obj *replace_op(node *op1, ARGS... args)
        {
            auto op2 = sketch<obj>(args...);
            // note that the new op can't be used by the old one,
            // so we only need to consider the case that new created
            // one takes the old one as input.

            bool cycleUse = false;
            auto &users = op1->get_user_handles();
            auto iter = std::find_if(users.begin(), users.end(), [op2](edgeHandle &n)
                                     {  
                if(!n) return false;
                return n->get_dual_node() == op2; });
            if (iter != users.end())
            {
                op2->add_output_edge(*iter);
                cycleUse = true;
            }
            op1->replace_by(op2);
            if (cycleUse)
            {
                insert_op(op2);
            }
            else
            {
                auto &nodes = point.g->get_nodes();
                auto iter = std::find(nodes.begin(), nodes.end(), op1);
                *iter = op2;
                op1->erase();
            }
            return op2;
        }

        //-----------------refactor marker -------------------

        template <typename origOp, typename targetOp>
        targetOp *isomorphic_rewrite(origOp *op)
        {
            auto newop = new targetOp();
            newop->set_parent_graph(op->get_parent_graph());
            newop->register_inputs(op->get_inputs());
            op->replace_by(newop);

            auto &nodes = op->get_parent_graph()->get_nodes();
            auto iter = std::find(nodes.begin(), nodes.end(), op);
            *iter = newop;
            return newop;
        }

        void goto_graph(graph *reg_)
        {
            point.g = reg_;
            point.nodes = &(reg_->get_nodes());
            point.iter = point.nodes->end();
        }

        paintPoint get_paintPoint() { return point; }

        void gotoParentGraph()
        {
            if (!point.g)
                return;
            goto_graph(point.g->get_parent_graph());
        }

        graph *get_parent_graph()
        {
            if (!point.g)
                return nullptr;
            return point.g->get_parent_graph();
        }

        LGFContext *get_context() { return ctx; }
        paintPoint point;
        LGFContext *ctx = nullptr;
    };

}

#endif