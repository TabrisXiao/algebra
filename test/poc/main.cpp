#include "graph/operation.h"
#include "graph/builtin.h"
#include "graph/builder.h"
#include "qft/math.h"
#include <iostream>

int main()
{
  using namespace lgf;
  using namespace lgf::math;
  utils::logger::get().set_level(utils::LOG_INFO);
  // Create a context
  context ctx;

  builder opb(&ctx);

  // Create a module operation
  auto module = opb.create<moduleOp>();

  // Create a define operation

  description desc = variable::get();
  opb.goto_region(module->get_region());
  auto define = opb.create<defineOp>(desc);
  auto def2 = opb.create<defineOp>(desc);
  auto add = opb.create<sumOp>(desc, &define->output(0), &def2->output(0));
  std::cout << "created" << std::endl;
  // Print the representation of the module operation
  module->get_region()->assign_sid();
  std::cout << module->represent() << std::endl;
  return 0;
}