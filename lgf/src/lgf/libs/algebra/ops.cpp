
#include "libs/algebra/ops.h"

namespace lgf
{

    resultCode productOp::rewrite(painter p, node *op)
    {
        if (op->get_input_size() == 1)
        {
            p.replace_op(op, op->input(0));
            return resultCode::success();
        }
        size_t i = 0;
        while( i < op->get_input_size()){
            auto node = op->input(i);
            i++;
            if(auto product = dynamic_cast<productOp*>(node)){
                auto new_inputs = product->get_input_nodes();
                op->drop_input(product);
                op->register_inputs_at(new_inputs, i);
                i++;
            }
        }
        return resultCode::pass();
    }
} // namespace lgf