#ifndef LGF_GROUP_H
#define LGF_GROUP_H
#include <memory.h>
#include "node.h"
#include "painter.h"
#include "pass.h"

namespace lgf
{

    class group
    {
    public:
        group() = default;
        virtual resultCode rewrite(painter &, node *op) = 0;
    };

    template <typename groupType>
    class groupRewriter : public rewriterBase
    {
    public:
        groupRewriter() = default;
        virtual resultCode execute(painter &rewriter, node *op) override final
        {
            if (auto g = dynamic_cast<groupType *>(op))
            {
                auto sig = g->rewrite(rewriter, op);
                return sig;
            }
            return resultCode::pass();
        }
    };

    class graphOperation : public group
    {
    public:
        graphOperation() = default;
    };

    class execGraphOpPass : public passBase
    {
    public:
        execGraphOpPass() : passBase("executeGOpPass") {}
        virtual resultCode run()
        {
            painter p(get_graph());
            add_rewriter<groupRewriter<graphOperation>>();
            return apply_rewriter_greedy(p, get_graph());
        }
    };

    class normalizer : public group
    {
    public:
        normalizer() = default;
    };
}

#endif