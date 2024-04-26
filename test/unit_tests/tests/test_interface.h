
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
      variable x, sig;
      auto y = function::exp(x / sig);
      auto z = function::power(x, 3.5);
      auto dy = function::d(y);
      auto dz = function::d(z);
      dz.latex();
      canvas::get()
          .get_pass_manager()
          .set_log_level(1);
      canvas::get().compile();
      return 0;
    }
  };
} // namespace test_body