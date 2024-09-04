
#ifndef LGF_PASS_H_
#define LGF_PASS_H_
#include "node.h"
#include "painter.h"
#include "utils.h"

namespace lgf
{

    class rewriterBase
    {
    public:
        rewriterBase() = default;
        virtual ~rewriterBase() = default;
        virtual resultCode execute(painter &, node *op) = 0;
        LGFContext *get_context() { return ctx; }
        LGFContext *ctx = nullptr;
    };

    template <typename concreteOp>
    class rewriter : public rewriterBase
    {
    public:
        rewriter() = default;
        virtual resultCode rewrite(painter &, concreteOp *op) = 0;
        virtual resultCode execute(painter &rewriter, node *op) override final
        {
            if (auto cop = dynamic_cast<concreteOp *>(op))
            {
                auto sig = rewrite(rewriter, cop);
                return sig;
            }
            return resultCode::pass();
        }
    };

    class normalizer
    {
    public:
        normalizer() = default;
        virtual ~normalizer() = default;
        virtual resultCode normalize(painter &, node *op) = 0;
    };

    class normalizeRewriter : public rewriterBase
    {
    public:
        normalizeRewriter() = default;
        virtual resultCode execute(painter &rewriter, node *op) final
        {
            if (auto cop = dynamic_cast<normalizer *>(op))
            {
                auto sig = cop->normalize(rewriter, op);
                return sig;
            }
            return resultCode::pass();
        }
    };

    class identifier
    {
    public:
        identifier() = default;
        virtual ~identifier() = default;
        symbolID get_uid_for_tree(node *op)
        {
            std::string id = op->get_sid();
            id += '(';
            std::vector<std::string> vec;
            for (auto i = 0; i < op->get_input_size(); i++)
            {
                if (auto n = dynamic_cast<identifier *>(op->input(i)))
                    vec.push_back(n->get_uid().value());
            }
            if (op->is_commutable())
            {
                std::sort(vec.begin(), vec.end());
            }
            for (auto i = 0; i < vec.size(); i++)
            {
                id = id + vec[i] + ',';
            }
            id.pop_back();
            id += ')';
            symbolID result = id;
            return result;
        }

        // get_uid: get unique id for the node this interface associated with.
        // if two nodes have the same uid, they are equivalent and should be removed by reduce process.
        virtual symbolID get_uid()
        {
            return get_uid_for_tree(dynamic_cast<node *>(this));
        }

        // static function can be used to test if two nodes are equivalent.
        static bool is_equivalent(node *a, node *b)
        {
            if (auto ia = dynamic_cast<identifier *>(a))
            {
                if (auto ib = dynamic_cast<identifier *>(b))
                {
                    return ia->get_uid() == ib->get_uid();
                }
            }
            return 0;
        }
    };

    class passBase
    {
    public:
        passBase(const char *name) : _pass_name(name) {}
        virtual ~passBase() = default;
        // the return value is not defined yet.
        virtual resultCode run() = 0;
        region *get_region() { return g; }
        void set_work_region(region *op) { g = op; }
        // addRewriter will create a rewriter using the arguments;
        template <typename T, typename... ARGS>
        void add_rewriter(ARGS... arg)
        {
            std::unique_ptr<rewriterBase> ptr = std::make_unique<T>(arg...);
            rewriters.push_back(std::move(ptr));
        }

        resultCode apply_reduce_once(painter &p, region *g);

        resultCode apply_rewriter_once(painter &p, region *g);
        resultCode apply_rewriter_greedy(painter &p, region *g);

        resultCode apply_rewriter_and_reduce_greedy(painter &p, region *g);

        resultCode walk_apply_rewriter_once(painter &p, region *g, bool deepwalk = 0);

        // Translation is a special method to apply rewriters,
        // It walk only once through a region in the dependency order
        // and apply all the applicable rewriters to the ops.
        // So this method is only safe for the case that all rewriters
        // are order free.
        bool translation(painter &p, region *g);

    public:
        std::vector<std::unique_ptr<rewriterBase>> rewriters;
        std::string _pass_name;
        bool rewriteHappen = 0;
        region *g = nullptr;
    };

    class passManager
    {
    public:
        passManager() = default;
        passManager(region *op) { reg = op; }
        void set_work_region(region *op) { reg = op; }
        void enable_print_after_pass() { bPrintAfterPass = 1; }
        void enable_print_before_pass() { bPrintBeforePass = 1; }
        void set_log_level(int level = 0)
        {
            if (level == 0)
            {
                bPrintAfterPass = 0;
                bPrintBeforePass = 0;
                bPrintInitialIR = 0;
                bPrintFinalIR = 0;
            }
            if (level > 0)
            {
                bPrintFinalIR = 1;
                bPrintInitialIR = 1;
            }
            if (level > 1)
            {
                bPrintAfterPass = 1;
            }
            if (level > 2)
            {
                bPrintBeforePass = 1;
            }
            if (level > 3)
            {
                bPrintForEachStep = 1;
            }
        }
        void init(region *op) { reg = op; }
        void validation(region *g);

        void run()
        {
            if (bPrintInitialIR)
            {
                OSTREAM << "\n------ Input " << name << " ------\n";
                reg->print();
                OSTREAM << "\n";
            }

            for (auto pass = passes.begin(); pass != passes.end(); pass++)
            {
                if (bPrintBeforePass)
                {
                    OSTREAM << "------ IR before pass:  " << (*pass).get()->_pass_name << " ------\n";
                    reg->print();
                    OSTREAM << "\n";
                }

                (*pass)->run();
                validation(reg);
                if (bPrintAfterPass)
                {
                    OSTREAM << "------ IR after pass: " << (*pass).get()->_pass_name << " ------\n";
                    reg->print();
                    OSTREAM << "\n";
                }
            }
            if (bPrintFinalIR)
            {
                OSTREAM << "\n------ IR after " << name << " ------\n";
                reg->print();
                OSTREAM << "\n";
            }
        }

        void add_pass(std::unique_ptr<passBase> ps, region *g = nullptr)
        {
            if (!g)
                g = reg;
            ps->set_work_region(dynamic_cast<region *>(g));
            passes.push_back(std::move(ps));
        }
        void flush()
        {
            passes.clear();
        }
        std::vector<std::unique_ptr<passBase>> passes;

        bool bPrintAfterPass = 0;
        bool bPrintBeforePass = 0;
        bool bPrintInitialIR = 0;
        bool bPrintFinalIR = 0;
        bool bPrintForEachStep = 0;
        region *reg = nullptr;
        std::string name = "";
    };

}

#endif