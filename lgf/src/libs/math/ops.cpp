#include "libs/builtin/ops.h"
#include "libs/math/ops.h"
#include "libs/math/util.h"

namespace lgf::math
{
  resultCode sumOp::normalize(painter &p, node *op)
  {
    if (op->get_input_size() == 1)
    {
      p.replace_op(op, op->input(0));
      return resultCode::success();
    }
    resultCode result = resultCode::pass();
    result.add(flatten_same_type_inputs<sumOp>(op));
    return result;
  };

  resultCode productOp::normalize(painter &p, node *op)
  {
    resultCode result = resultCode::pass();
    if (op->get_input_size() == 1)
    {
      p.replace_op(op, op->input(0));
      return resultCode::success();
    }
    auto g = op->get_parent_region();

    // g->print();
    result.add(flatten_same_type_inputs<productOp>(op));

    return result;
  }

  resultCode funcExponentationOp::normalize(painter &p, node *op)
  {

    auto base = op->input(0);
    auto exp = base->dyn_cast<funcExponentationOp>();
    if (exp)
    {
      // merge (a^x)^y to a^(x*y)
      auto power = exp->input(1);
      auto new_exp = p.replace_op<funcExponentationOp>(op, exp->input(0), p.paint<productOp>(exp->input(1), op->input(1)));
      return resultCode::success();
    }

    return resultCode::pass();
  }
  resultCode funcLogarithmOp::normalize(painter &p, node *op)
  {
    auto ctx = p.get_context();
    auto log = op->dyn_cast<funcLogarithmOp>();
    auto base = log->base();
    auto arg = log->arg();
    if (base == arg)
    {
      auto unit = realNumber::get();
      auto unitAttr = realNumberData::get(realNumberData::real, 1);
      p.replace_op<lgf::cstDeclOp>(op, unit, unitAttr);
      return resultCode::success();
    }
    return resultCode::pass();
  }
} // namespace lgf