#include "libs/math/functional/passes.h"

std::unique_ptr<lgf::passBase> lgf::createCalculusPass()
{
    return std::make_unique<CalculusPass>();
}