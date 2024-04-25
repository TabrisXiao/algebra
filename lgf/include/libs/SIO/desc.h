
#ifndef LGF_LIB_SIO_DESC_H
#define LGF_LIB_SIO_DESC_H
#include "lgf/value.h"
#include "lgf/context.h"

namespace lgf::sio
{
    class expDesc : public simpleValue
    {
    public:
        expDesc(LGFContext *ctx) : simpleValue("expressible") {}
    };
} // namespace lgf::sio

#endif