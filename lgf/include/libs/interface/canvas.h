
#ifndef LGF_INTERFACE_CANVAS_H
#define LGF_INTERFACE_CANVAS_H
#include "lgf/context.h"
#include "lgf/painter.h"
#include "libs/builtin/builtin.h"
#include "libs/sio/exporter.h"
#include "libs/math/functional/passes.h"
#include "libs/transform/convertToSIO.h"

namespace lgi
{

  // canvas is a singleton class that contains the information about
  // painter and context for lgf that currently working.
  class canvas
  {
  public:
    canvas(canvas &) = delete;
    ~canvas() = default;
    void operator=(const canvas &) = delete;

    static canvas &get()
    {
      if (gcanvas != nullptr)
        return *gcanvas;
      gcanvas = new canvas();
      return *gcanvas;
    }
    static void restart()
    {
      if (gcanvas)
        delete gcanvas;
      get();
    }
    static void start() { get(); }
    static void end()
    {
      if (gcanvas)
      {
        delete gcanvas;
      }
    }

    lgf::painter &get_painter() { return p; }
    lgf::LGFContext &get_context() { return ctx; }
    void print() { g.print(); }

    void compile()
    {
      pm.set_work_region(&g);
      pm.add_pass(lgf::math::createCalculusPass());
      // pm.add_pass(lgf::createAlgebraNormalizationPass());
      pm.name = "compile";
      pm.run();

      export2latex();
    }

    void export2latex()
    {
      pm.flush();
      pm.name = "export";
      pm.add_pass(lgf::transform::createConvertToSIOPass());
      pm.run();
      lgf::sio::export2latex ex(&g);
      g.assign_id();
      ex.run_on_op();
    }

    lgf::passManager &get_pass_manager() { return pm; }

  protected:
    canvas()
    {
      p.set_context(&ctx);
      p.goto_graph(&g);
    }
    inline static canvas *gcanvas = nullptr;
    lgf::moduleOp g;
    lgf::LGFContext ctx;
    lgf::painter p;
    lgf::passManager pm;
  };

} // namespace lgi

#endif