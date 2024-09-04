
#ifndef LGF_BUILTIN_PASSES_H
#define LGF_BUILTIN_PASSES_H
#include "lgf/pass.h"
#include "ops.h"

namespace lgf
{

    class normalizationPass : public passBase
    {
    public:
        normalizationPass(const char *name = "normalization") : passBase(name) {}
        virtual resultCode run()
        {
            painter p(get_region());
            add_rewriter<normalizeRewriter>();
            // remove_unused_ops(get_region());
            remove_identical_ops(p, get_region());
            resultCode code = apply_rewriter_greedy(p, get_region());
            remove_unused_ops(get_region());
            remove_identical_ops(p, get_region());
            get_region()->clean();
            return code;
        }

        void remove_trivial_op(node *op);

        void remove_unused_ops(region *g);

        bool remove_identical_ops(painter p, region *g);
    };

    std::unique_ptr<passBase>
    createNormalizationPass();
} // namespace lgf

#endif