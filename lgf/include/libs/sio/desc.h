
#ifndef LGF_LIB_SIO_DESC_H
#define LGF_LIB_SIO_DESC_H
#include "lgf/value.h"
#include "lgf/context.h"

namespace lgf::sio
{
    class expDesc : public descBase
    {
    public:
        expDesc() : descBase("expressible") {}
        static descriptor get()
        {
            return descriptor::get<expDesc>();
        }
        std::unique_ptr<descBase> copy()
        {
            return std::make_unique<expDesc>();
        }
    };
} // namespace lgf::sio

#endif