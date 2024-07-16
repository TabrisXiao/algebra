
#include "lgf/pass.h"
#include "lgf/group.h"
// #include "utility.h"
using namespace lgf;

resultCode passBase::apply_rewriter_once(painter &p, graph *g)
{
  resultCode result;
  p.goto_graph(g);
  for (auto ptr = rewriters.begin(); ptr != rewriters.end(); ptr++)
  {
    std::vector<node *> nodes = g->get_nodes();
    for (auto i = 0; i < nodes.size(); i++)
    {
      auto &node = nodes[i];
      if (node->is_deprecate())
        continue;
      p.set_paintPoint_after(node);
      result.add((*ptr).get()->execute(p, node));
      if (node->is_deprecate())
        continue;
      if (auto subg = dynamic_cast<graph *>(node))
      {
        result.add(apply_rewriter_once(p, subg));
      }
    }
  }
  return result;
}
//---------------------------------------------------

resultCode passBase::apply_rewriter_greedy(painter &p, graph *g)
{
  auto result = apply_rewriter_once(p, g);
  g->clean();
  int counts = 1;
  auto final_result = result;
  while (result.isSuccess())
  {
    result = apply_rewriter_once(p, g);
    if (0)
    {
      OSTREAM << "\n------ Pass: " << _pass_name << " step: " << counts
              << " ------\n";
      g->print();
      OSTREAM << "\n";
    }
    counts++;
    final_result.add(result);
    g->clean();
  }
  return final_result;
}
//---------------------------------------------------

resultCode passBase::walk_apply_rewriter_once(painter &p, graph *g,
                                              bool deepWalk)
{
  p.goto_graph(g);
  resultCode result;
  g->walk(
      [&](node *op)
      {
        if (op->is_deprecate())
          return;
        for (auto ptr = rewriters.begin(); ptr != rewriters.end(); ptr++)
        {
          result.add((*ptr).get()->execute(p, op));
        }
      },
      deepWalk);
  return result;
}

//---------------------------------------------------
resultCode passBase::apply_reduce_once(painter &p, graph *g)
{
  resultCode result;
  p.goto_graph(g);
  g->assign_id();
  std::queue<node *> q;
  for (auto op : g->get_nodes())
  {
    if (op->is_deprecate() || op->get_input_size() != 0 || dynamic_cast<identiferInterface *>(op) == nullptr)
      continue;
    q.push(op);
  }

  while (q.size())
  {
    auto top = q.front();
    top->set_exploration(true);
    q.pop();
    if (top->is_deprecate())
      continue;
    auto id = top->get_sid();

    // checking if current op has the same uid with the rest nodes in the graph.
    // if so, replace the later op by current op.
    for (auto op : g->get_nodes())
    {
      if (op == top || op->is_deprecate() || op->get_sid() != id)
        continue;
      auto cuid = op->dyn_cast<identiferInterface>()->get_uid();
      if (cuid == id)
      {
        p.replace_op(op, top);
        result.add(resultCode::success());
      }
    }

    // adding users of current op into the queue
    for (auto &h : top->get_user_handles())
    {
      auto user = h.get_dual_node();
      if (user->is_deprecate() || user->is_explored() || user->isa<identiferInterface>())
        continue;
      q.push(user);
    }
  }
  g->reset_walk_status();
  return result;
}

//---------------------------------------------------
resultCode passBase::apply_rewriter_and_reduce_greedy(painter &p, graph *g)
{
  auto result = apply_rewriter_once(p, g);
  result.add(apply_reduce_once(p, g));
  g->clean();
  int counts = 1;
  auto final_result = result;
  while (result.isSuccess())
  {
    result = apply_rewriter_once(p, g);
    result.add(apply_reduce_once(p, g));
    if (0)
    {
      OSTREAM << "\n------ Pass: " << _pass_name << " step: " << counts
              << " ------\n";
      g->print();
      OSTREAM << "\n";
    }
    counts++;
    final_result.add(result);
    g->clean();
  }
  return final_result;
}

//---------------------------------------------------
bool passBase::translation(painter &p, graph *g)
{
  g->walk(
      [&](node *op)
      {
        if (op->is_deprecate())
          return;
        for (auto ptr = rewriters.begin(); ptr != rewriters.end(); ptr++)
        {
          (*ptr).get()->execute(p, op);
        }
      },
      0);
  return 0;
}
//---------------------------------------------------

void passManager::validation(graph *g)
{
  auto nodes = g->get_nodes();
  for (auto &node : nodes)
  {
    if (node->is_deprecate())
      continue;
    if (auto subg = dynamic_cast<graph *>(node))
    {
      validation(subg);
    }
  }
  if (g->clean())
  {
    validation(g);
  }
}
//---------------------------------------------------
