#ifndef LGF_GROUP_H
#define LGF_GROUP_H
#include <memory.h>
#include "node.h"
#include "painter.h"
#include "pass.h"

namespace lgf
{

    class group
    {
    public:
        group() = default;
        virtual resultCode rewrite(painter, node *op) = 0;
    };

    template <typename groupType>
    class groupRewriter : public rewriterBase
    {
    public:
        groupRewriter() = default;
        virtual resultCode execute(painter rewriter, node *op) override final
        {
            if (auto g = dynamic_cast<groupType *>(op))
            {
                auto sig = g->rewrite(rewriter, op);
                return sig;
            }
            return resultCode::pass();
        }
    };

    class graphOperation : public group
    {
    public:
        graphOperation() = default;
    };

    class execGraphOpPass : public passBase
    {
    public:
        execGraphOpPass() : passBase("executeGOpPass") {}
        virtual resultCode run()
        {
            painter p(get_graph());
            add_rewriter<groupRewriter<graphOperation>>();
            return apply_rewriter_greedy(p, get_graph());
        }
    };

    class normalizer : public group
    {
    public:
        normalizer() = default;
    };

    class normalizationPass : public passBase
    {
    public:
        normalizationPass() : passBase("normalization") {}
        virtual resultCode run()
        {
            painter p(get_graph());
            add_rewriter<groupRewriter<normalizer>>();
            // applyRewriterOnce(p, getGraph());
            // return applyRewriterOnce(p, getGraph());
            remove_identical_ops(p, get_graph());
            resultCode code = apply_rewriter_greedy(p, get_graph());
            remove_unused_ops(get_graph());
            get_graph()->clean();
            return code;
        }

        void remove_trivial_op(node *op)
        {
            if (op->is_deprecate())
                return;
            if (op->is_trivial())
            {
                for (auto &h : op->get_user_handles())
                {
                    if (h && h->is_coupled())
                    {
                        auto node = h->get_dual_node();
                        remove_trivial_op(node);
                    }
                }
                if (!op->get_user_size())
                {
                    op->erase();
                }
            }
        }

        void remove_unused_ops(graph *g)
        {
            for (auto &op : g->get_nodes())
            {
                if (auto subg = dynamic_cast<graph *>(op))
                {
                    remove_unused_ops(subg);
                }
                else
                    remove_trivial_op(op);
            }
        }

        bool remove_identical_ops(painter p, graph *g)
        {
            // using breadth first walk to remove identical ops
            // to avoid the case that replace of the early ops makes the later
            // ops identical.
            // assignID is necessary for the checkIfIdenticalExist function as the id is used to check if two ops are identical

            bool changed = false;
            if (!g)
                return false;
            g->assign_id(0);

            auto list = g->get_nodes();
            std::queue<node *> queue;
            for (auto op : list)
            {
                if (op->is_deprecate())
                    continue;
                if (op->get_input_size() != 0)
                    continue;
                // if op is a graph:

                queue.push(op);
                // if( !checkIfIdenticalExist(op, queue) ){
                //     queue.push(op);
                // }
            }

            while (queue.size() > 1)
            {
                auto op = queue.front();
                queue.pop();
                op->set_exploration(true);
                if (auto subg = dynamic_cast<graph *>(op))
                {
                    painter pp(subg);
                    remove_identical_ops(pp, subg);
                    continue;
                }
                // checking if the duplicate op exists and
                // replace it if so.
                auto mark = queue.front();
                auto target = op->get_op_represent();
                do
                {
                    auto checkop = queue.front();
                    queue.pop();
                    if (checkop != op && target == checkop->get_op_represent())
                    {
                        if (mark == checkop)
                            mark = queue.front();
                        // std::cout<<"placing: "<<checkop->represent()<<" by "<<op->represent()<<std::endl;
                        checkop->replace_by(op);
                        checkop->erase();
                        changed = true;
                    }
                    else
                    {
                        queue.push(checkop);
                    }
                } while (queue.front() != mark);

                for (auto &h : op->get_user_handles())
                {
                    if (!h->is_coupled())
                        continue;
                    auto user = h->get_dual_node();
                    if (user->is_deprecate() || user->is_explored() || !user->is_dependency_fullfilled())
                        continue;
                    queue.push(user);
                }
            }
            for (auto node : list)
            {
                node->set_exploration(false);
            }
            return changed;
        }
    };

}

#endif