#include "tests/test_builtin.h"
#include "tests/test_interface.h"
#include "tests/test_node.h"
#include "unit_test_frame.h"
#include <iostream>
// #include "tests/test_clean_pass.h"
// #include "tests/test_fa.h"
// #include "tests/test_stat.h"
// #include "test_codegen.h"

int main()
{
  using namespace test_body;
  unit_test_frame frame;
  // frame.add_test<test_node>();
  frame.add_test<test_builtin>();
  frame.add_test<test_interface>();
  frame.run_all_test();
  return 0;
}