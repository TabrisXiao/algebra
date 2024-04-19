

#ifndef LIBS_SIO_EXPORT_H
#define LIBS_SIO_EXPORT_H

#include "lgf/global.h"
#include "lgf/node.h"
#include "ops.h"

namespace lgf::SIO
{
    class exporterBase
    {
    public:
        exporterBase(graph *g_) : g(g_) {}

        virtual std::string process(node *val) = 0;

        void run(graph *entry);
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
                os << process(node->input());
                p.replace_op(node, node->input());
                node->erase();
            }
            os << "\n--------- end ---------\n\n";
        }

        global::stream &os = global::stream::getInstance();
        graph *g = nullptr;
    };

    class export2latex : public exporterBase
    {
    public:
        export2latex(graph *g_) : exporterBase(g_) {}

        virtual std::string process(node *val) final;

        void run(graph *entry){};
        void run() { run(g); }
        void run_on_op()
        {
            exporterBase::run_on_op<latexExportOp>();
        }
    };
} // namespace lgf::SIO

#endif