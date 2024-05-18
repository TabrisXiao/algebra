
#ifndef LGF_BUILTIN_PASSES_H
#define LGF_BUILTIN_PASSES_H
#include "lgf/group.h"
#include "ops.h"

namespace lgf
{

    class normalizationPass : public passBase
    {
    public:
        normalizationPass(const char *name = "normalization") : passBase(name) {}
        virtual resultCode run()
        {
            painter p(get_graph());
            add_rewriter<groupRewriter<normalizer>>();
            //remove_unused_ops(get_graph());
            remove_identical_ops(p, get_graph());
            resultCode code = apply_rewriter_greedy(p, get_graph());
            remove_unused_ops(get_graph());
            remove_identical_ops(p, get_graph());
            get_graph()->clean();
            return code;
        }

        void remove_trivial_op(node *op);

        void remove_unused_ops(graph *g);

        bool remove_identical_ops(painter p, graph *g);
    };

    std::unique_ptr<passBase>
    createNormalizationPass();
} // namespace lgf

#endif