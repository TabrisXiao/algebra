
#include "lgf/node.h"
#include "lgf/pass.h"
#include "libs/builtin/passes.h"

namespace lgf
{
    std::unique_ptr<passBase> createNormalizationPass()
    {
        return std::make_unique<normalizationPass>();
    }

    void normalizationPass::remove_trivial_op(node *op)
    {
        if (op->is_deprecate())
            return;
        if (!op->get_status(eNonTrivial))
        {
            for (auto &h : op->get_user_handles())
            {
                if (h.is_coupled())
                {
                    auto node = h.get_dual_node();
                    remove_trivial_op(node);
                }
            }
            if (!op->get_user_size())
            {
                op->erase();
            }
        }
    }

    void normalizationPass::remove_unused_ops(graph *g)
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

    bool normalizationPass::remove_identical_ops(painter p, graph *g)
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
            queue.push(op);
        }

        while (queue.size())
        {
            auto op = queue.front();
            queue.pop();
            while (queue.size() && op == queue.front())
            {
                queue.pop();
            }
            op->set_exploration(true);
            // skip non-trivial op
            if (op->get_status(eNonTrivial))
            {
                continue;
            }
            // checking if the duplicate op exists and
            // replace it if so.
            lgf::node *mark = nullptr;
            while (queue.size() > 0)
            {
                if (queue.front()->is_deprecate())
                {
                    queue.pop();
                    continue;
                }
                mark = queue.front();
                break;
            }
            if (mark)
            {
                auto target = op->get_op_represent();
                do
                {
                    if (!mark)
                        mark = queue.front();
                    auto checkop = queue.front();
                    queue.pop();
                    if (checkop == op || checkop->is_deprecate())
                        continue;
                    if (target == checkop->get_op_represent())
                    {
                        checkop->replace_by(op);
                        checkop->erase();
                        changed = true;
                        if (mark == checkop)
                            mark = nullptr;
                    }
                    else
                    {
                        queue.push(checkop);
                    }
                } while (queue.front() != mark && !queue.empty());
            }
            for (auto &h : op->get_user_handles())
            {
                if (!h.is_coupled())
                    continue;
                auto user = h.get_dual_node();
                if (user->is_deprecate() || user->is_explored() || !user->is_dependency_fullfilled())
                    continue;
                if (!user->get_status(eNonTrivial))
                {
                    queue.push(user);
                }
            }
        }
        for (auto node : list)
        {
            node->set_exploration(false);
        }
        return changed;
    }
} // namespace lgf
