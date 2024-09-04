
#ifndef LGF_PAINTER_H_
#define LGF_PAINTER_H_
#include "global.h"
#include "node.h"
#include <algorithm>
#include <iostream>
#include <memory.h>
#include <unordered_set>

// abstract node region

namespace lgf
{

  class rewriterBase;

  class LGFContext;

  class painter
  {
  public:
    struct paintPoint
    {
      region *reg = nullptr;
      std::vector<node *> *nodes = nullptr;
      std::vector<node *>::iterator iter = std::vector<node *>::iterator();

      bool is_invalid()
      {
        if (reg == nullptr || nodes == nullptr)
          return true;
        if (iter < nodes->begin() || iter > nodes->end())
          return true;
        return false;
      }
    };
    painter() = default;
    painter(region *g)
        : point({g, &(g->get_nodes()), g->get_nodes().end()}),
          ctx(g->get_parent()->get_context()) {}
    painter(painter &p) : point(p.get_paintPoint()), ctx(p.get_context()) {}
    ~painter() {}
    void set_context(LGFContext *ctx_) { ctx = ctx_; }

    void set_paintPoint_to_region_begin()
    {
      point.iter = point.reg->get_nodes().begin();
    }
    void set_paintPoint_before(node *op)
    {
      point.reg = op->get_parent_region();
      auto &vec = point.reg->get_nodes();
      point.iter = std::find(vec.begin(), vec.end(), op);
      if (point.iter != vec.begin())
        point.iter--;
    }
    void set_paintPoint_after(node *op)
    {
      point.reg = op->get_parent_region();
      auto &vec = point.reg->get_nodes();
      point.iter = std::find(vec.begin(), vec.end(), op);
      if (point.iter != vec.end())
        point.iter++;
    }

    void set_paintPoint_at(region *g, std::vector<node *> *vec,
                           std::vector<node *>::iterator iter)
    {
      point.reg = g;
      point.nodes = vec;
      point.iter = iter;
    }

    void reset_paintPoint()
    {
      if (!point.reg)
        return;
      point.iter = point.reg->get_nodes().end();
    }

    template <typename obj>
    obj *sketch()
    {
      auto op = obj::build(ctx);
      // op->inferType(ctx);
      return op;
    }

    template <typename obj, typename... ARGS>
    obj *sketch(ARGS... args)
    {
      auto op = obj::build(ctx, args...);
      // op->inferType(ctx);
      return op;
    }

    template <typename obj, typename... ARGS>
    obj *paint(ARGS... args)
    {
      THROW_WHEN(point.is_invalid(), "paint point is invalid!")
      // CHECK_CONDITION(point.reg!=nullptr, "No region associated to the painter!");
      auto op = sketch<obj>(args...);
      // add to region
      op->set_parent_region(point.reg);
      point.iter = point.nodes->insert(point.iter, op) + 1;
      return op;
    }

    template <typename obj>
    obj *paint()
    {
      THROW_WHEN(point.is_invalid(), "paint point is invalid!")
      auto op = sketch<obj>();
      // add to region
      op->set_parent_region(point.reg);
      point.iter = point.nodes->insert(point.iter, op) + 1;
      return op;
    }

    // void insert_op(node *op) {
    //   point.iter = point.nodes->insert(point.iter, op) + 1;
    // }

    template <typename obj>
    obj *replace_op(node *op1)
    {
      auto op2 = sketch<obj>();
      // There's no input for op1 so we can just replace
      // them directly
      op1->replace_by(op2);
      auto &nodes = point.reg->get_nodes();
      auto iter = std::find(nodes.begin(), nodes.end(), op1);
      (*iter) = op2;
      op1->erase();
    }

    void replace_op(node *op1, node *op2)
    {
      // the replace function assume that both ops are in
      // the same region
      if (op1 == op2)
        return;

      if (op2->get_parent_region() && op2->get_parent_region() != point.reg)
      {
        throw std::runtime_error("replace_op: ops are in different regions!");
      }
      // because the op1 is gonna replaced by op2, so
      // we assume that op1 can't be used by op2
      op1->replace_by(op2);
      auto iter = std::find(point.nodes->begin(), point.nodes->end(), op1);
      bool keepOrigOp = false;
      if (op1->get_user_size())
      {
        for (auto &h : op1->get_user_handles())
        {
          std::cout << "--remind user: " << h.get_dual_edge() << std::endl;
          keepOrigOp = 1;
        }
        keepOrigOp = 1;
      }

      if (iter != point.nodes->end() && !keepOrigOp)
      {
        if (op2->get_parent_region() != point.reg)
        {
          *iter = op2;
        }
        op1->erase();
      }
      else
      {
        if (op2->get_parent_region() != point.reg)
          point.iter = point.nodes->insert(iter + 1, op2) + 1;
        if (!keepOrigOp)
        {
          op1->erase();
        }
      }
      op2->set_parent_region(point.reg);
    }

    template <typename obj, typename... ARGS>
    obj *replace_op(node *op1, ARGS... args)
    {
      auto op2 = sketch<obj>(args...);
      // note that the new op can't be used by the old one,
      // so we only need to consider the case that new created
      // one takes the old one as input.

      replace_op(op1, op2);
      return op2;
    }

    //-----------------refactor marker -------------------

    template <typename origOp, typename targetOp>
    targetOp *isomorphic_rewrite(origOp *op)
    {
      auto newop = new targetOp();
      newop->set_parent_region(op->get_parent_region());
      for (auto &h : op->get_input_handles())
      {
        newop->register_input(h.get_link_node());
      }

      op->replace_by(newop);

      auto &nodes = op->get_parent_region()->get_nodes();
      auto iter = std::find(nodes.begin(), nodes.end(), op);
      *iter = newop;
      op->erase();
      return newop;
    }

    void goto_region(region *reg_)
    {
      point.reg = reg_;
      point.nodes = &(reg_->get_nodes());
      point.iter = point.nodes->end();
    }

    paintPoint get_paintPoint() { return point; }

    // void gotoParentregion()
    // {
    //   if (!point.reg)
    //     return;
    //   goto_region(point.reg->get_parent_region());
    // }

    // region *get_parent_region()
    // {
    //   if (!point.reg)
    //     return nullptr;
    //   return point.reg->get_parent_region();
    // }

    LGFContext *get_context() { return ctx; }
    paintPoint point;
    LGFContext *ctx = nullptr;
  };

} // namespace lgf

#endif