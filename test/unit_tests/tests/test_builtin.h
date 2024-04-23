
#pragma once

#include "lgf/painter.h"
#include "libs/Builtin/Builtin.h"
#include "unit_test_frame.h"

namespace test_body
{
  using namespace lgf;
  class test_builtin : public test_wrapper
  {
  public:
    test_builtin() { test_id = "builtin lib test"; };
    bool run()
    {
      moduleOp module;
      painter p(&module);
      auto ctx = p.get_context();
      auto intV = ctx->get_desc<int32Value>();
      auto data = lgf::int32Data(3);
      auto cst = p.paint<lgf::cstDeclOp>(intV, &data);
      auto x = p.paint<lgf::declOp>(intV);
      auto assign = p.paint<lgf::updateOp>(x, cst);
      auto ret = p.paint<lgf::returnOp>(assign);

      auto floatV = ctx->get_desc<float32Value>();
      auto fdsc = ctx->get_desc<funcDesc>(floatV, std::vector<valueDesc *>{intV});
      auto func = p.paint<lgf::funcDefineOp>("convert", fdsc);
      auto call = p.paint<lgf::funcCallOp>(func, x);
      module.print();
      return 0;
    }
  };
} // namespace test_body