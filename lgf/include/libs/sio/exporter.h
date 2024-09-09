

#ifndef LIBS_SIO_EXPORT_H
#define LIBS_SIO_EXPORT_H

#include "lgf/global.h"
#include "lgf/node.h"
#include "ops.h"

namespace lgf::sio
{
    class exporterBase
    {
    public:
        exporterBase(region *g_) : g(g_) {}

        virtual std::string process(node *val) = 0;

        void run(region *entry);
        void run() { run(g); }

        template <typename opTy>
        void run_on_op()
        {
            painter p(g);
            os << "\n\n--------- latex exports: ---------\n";
            for (auto &node : g->get_nodes())
            {
                if (!dynamic_cast<opTy *>(node))
                    continue;
                os << " --- \n";
                os << process(node->input());
                node->erase();
                os << "\n";
            }
            os << "\n--------- end ---------\n\n";
        }

        global::stream &os = global::stream::getInstance();
        region *g = nullptr;
    };

    class export2latex : public exporterBase
    {
    public:
        export2latex(region *g_) : exporterBase(g_) {}

        virtual std::string process(node *val) final;
        std::string simp_func_expression(node *val);
        void run(region *entry) {};
        void run() { run(g); }
        void run_on_op()
        {
            exporterBase::run_on_op<latexExportOp>();
        }
    };
} // namespace lgf::sio

#endif