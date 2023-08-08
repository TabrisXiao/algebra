
#ifndef LGF_MODULE_H
#define LGF_MODULE_H

#include "painter.h"

namespace lgf{

class LGFModule{
    public:
    LGFModule() = default;
    virtual void registerTypes() = 0;
    virtual void registerInitPass() {}
};

}

#endif