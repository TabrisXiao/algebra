
#include "libs/algebra/ops.h"
#include "libs/builtin/ops.h"

namespace lgf {

resultCode productOp::rewrite(painter &p, node *op) {
  resultCode result = resultCode::pass();
  if (op->get_input_size() == 1) {
    p.replace_op(op, op->input(0));
    return resultCode::success();
  }
  size_t i = 0;
  while (i < op->get_input_size()) {
    auto node = op->input(i);
    i++;
    if (auto product = dynamic_cast<productOp *>(node)) {
      auto new_inputs = product->get_input_nodes();
      op->drop_input(product);
      op->register_inputs_at(new_inputs, i);
      i++;
      result.add(resultCode::success());
    } else if (auto var = dynamic_cast<declOp *>(node)) {
      auto unit = var->get_value_desc_as<unitDesc>();
      if (!unit)
        continue;
      if (i > 1) {
        auto lhs = op->input(i - 2);
        if (unit->unit_effective_check(lhs->get_value_desc())) {
          op->drop_input(var);
          continue;
        }
      } else if (i < op->get_input_size()+1) {
        auto rhs = op->input(i);
        if (unit->unit_effective_check(rhs->get_value_desc())) {
          op->drop_input(var);
          continue;
        }
      }
    }
  }

  return result;
}
} // namespace lgf