
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
    auto g = op->get_parent_graph();

    // g->print();
    result.add(flatten_same_type_inputs<productOp>(op));

    return result;
  }
} // namespace lgf