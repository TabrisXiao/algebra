
#include "libs/algebra/ops.h"
#include "libs/builtin/ops.h"
#include "libs/algebra/util.h"

namespace lgf
{

  resultCode sumOp::rewrite(painter &p, node *op)
  {
    if (op->get_input_size() == 1)
    {
      p.replace_op(op, op->input(0));
      return resultCode::success();
    }
    resultCode result = resultCode::pass();
    result.add(flatten_same_type_inputs<sumOp>(op));
    result.add(run_op_pair_base_on_desc<zeroDesc, valueDesc>(op, [&op](node *lhs, node *rhs) -> resultCode
                                                             {
        op->drop_input(lhs);
        return resultCode::success();
      return resultCode::pass(); }));
    return result;
  };

  resultCode productOp::rewrite(painter &p, node *op)
  {
    resultCode result = resultCode::pass();
    if (op->get_input_size() == 1)
    {
      p.replace_op(op, op->input(0));
      return resultCode::success();
    }
    result.add(flatten_same_type_inputs<productOp>(op));
    result.add(run_op_pair_base_on_desc<unitDesc, valueDesc>(op, [&op](node *lhs, node *rhs) -> resultCode
                                                             {
      auto unit = lhs->get_value_desc_as<unitDesc>();
      auto value = rhs->get_value_desc();
      if (unit->unit_effective_check(value))
      {
        op->drop_input(lhs);
        return resultCode::success();
      } 
      return resultCode::pass(); }));

    result.add(run_op_pair_base_on_desc<zeroDesc, valueDesc>(op, [&op](node *lhs, node *rhs) -> resultCode
                                                             {
      auto zero = lhs->get_value_desc_as<zeroDesc>();
      auto value = rhs->get_value_desc();
      if (zero->zero_effective_check(value))
      {
        op->drop_input(rhs);
        return resultCode::success();
      } 
      return resultCode::pass(); }));

    return result;
  }
} // namespace lgf