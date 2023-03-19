
#include "aog.h"
#include "utility.h"
using namespace aog;

void opModifier::removeElement(element *e){
    auto op = e->getDefiningOp();
    auto & out_ops = op->getOutVertices();
    for(auto op_ : out_ops){
        auto op = dynamic_cast<operation*>(op_);
        utility::remove_value_from_vector<element*>(e, op->getInputs());
    }
}