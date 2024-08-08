#include "libs/math/functional/passes.h"

std::unique_ptr<lgf::passBase> lgf::math::createCalculusPass()
{
    return std::make_unique<CalculusPass>();
}