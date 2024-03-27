
#ifndef LGF_INTERFACE_CANVAS_H
#define LGF_INTERFACE_CANVAS_H
#include "lgf/LGFContext.h"
#include "lgf/painter.h"
#include "libs/Builtin/Builtin.h"

namespace lgi{

// canvas is a singleton class that contains the information about 
// painter and context for lgf that currently working.
class canvas {
    public:
    canvas(canvas &) = delete;
    ~canvas() = default;
    void operator=(const canvas &) = delete;

    static canvas &get(){
        if (gcanvas != nullptr)
            return *gcanvas;
        gcanvas = new canvas();
        return *gcanvas;
    }
    static void restart(){
        if(gcanvas) delete gcanvas;
        get();
    }
    static void start(){
        get();
    }
    static void end(){
        if(gcanvas) {
            delete gcanvas;
        }
    }
    
    lgf::painter & getPainter() {
        return p;
    }
    lgf::LGFContext &getContext(){
        return ctx;
    }
    void print(){
        g.print();
    }

protected:
    canvas(){
        p.setContext(&ctx);
        p.gotoGraph(&g);
    }
    inline static canvas *gcanvas = nullptr;
    lgf::moduleOp g;
    lgf::LGFContext ctx;
    lgf::painter p;

};

} // namespace  lgf

#endif