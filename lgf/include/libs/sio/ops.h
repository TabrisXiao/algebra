#ifndef MATH_SIO_OPS_H
#define MATH_SIO_OPS_H
#include "lgf/node.h"
#include "libs/builtin/builtin.h"
#include "libs/math/ops.h"

namespace lgf::sio
{
    class exportOp : public node
    {
    public:
        exportOp(std::string name) : node(name)
        {
            mark_status(eNonTrivial);
        }
        static exportOp *build(LGFContext *ctx, node *value)
        {
            exportOp *op = new exportOp("sio::export");
            op->register_input(value);
            return op;
        }
        virtual sid_t represent() override
        {
            return get_sid() + ": " + input()->value_rep();
        }
    };

    class latexExportOp : public exportOp
    {
    public:
        latexExportOp() : exportOp("sio::LatexExporter") {}
        static latexExportOp *build(LGFContext *ctx, node *value)
        {
            latexExportOp *op = new latexExportOp();
            op->register_input(value);
            return op;
        }
    };

    class representOp : public node
    {
    public:
        representOp(std::string name) : node(name) {}
    };

}

#endif