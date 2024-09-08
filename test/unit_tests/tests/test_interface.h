
#pragma once

#include "libs/interface/interface.h"
#include "unit_test_frame.h"

using namespace lgi::math;

namespace test_body
{

  class test_interface : public test_wrapper
  {
  public:
    test_interface() { test_id = "interface test"; };
    bool run()
    {
      variable x, sig, z = 2;
      auto y = cos(cos(cos(x)));
      y.latex();
      auto dy = d(y);
      dy.latex();
      auto iy = integral(dy, x, 0, 1);
      iy.latex();
      canvas::get()
          .get_pass_manager()
          .set_log_level(2);
      canvas::get().compile();
      return 0;
    }
  };
} // namespace test_body