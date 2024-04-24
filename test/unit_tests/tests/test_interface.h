
#pragma once

#include "libs/interface/interface.h"
#include "unit_test_frame.h"

using namespace lgi;

namespace test_body
{

  class test_interface : public test_wrapper
  {
  public:
    test_interface() { test_id = "interface test"; };
    bool run()
    {
      variable x, z = 3, s = 1;
      auto y = function::cos(x) + z;
      auto i = function::integral(y, x, 0, 1);
      y.latex();
      i.check();
      canvas::get()
          .get_pass_manager()
          .set_log_level(1);
      canvas::get().compile();
      return 0;
    }
  };
} // namespace test_body