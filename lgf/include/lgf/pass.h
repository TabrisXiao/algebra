
#ifndef LGF_PASS_H_
#define LGF_PASS_H_
#include "node.h"
#include "painter.h"
#include "utils.h"

namespace lgf
{

    class resultCode : public bitCode<int8_t>
    {
    public:
        enum result : int8_t
        {
            default_result,
            success_result,
            failed_result
        };
        resultCode() : bitCode() { value = 0; }
        resultCode(int8_t v) : bitCode(int8_t(v)) {}
        static resultCode success()
        {
            return resultCode(int8_t(resultCode::result::success_result));
        }
        static resultCode fail()
        {
            return resultCode(int8_t(resultCode::result::failed_result));
        }

        static resultCode pass()
        {
            return resultCode(int8_t(resultCode::result::default_result));
        }

        bool isSuccess()
        {
            return check(success_result);
        }
    };

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

    class passBase
    {
    public:
        passBase(const char *name) : _pass_name(name) {}

        // the return value is not defined yet.
        virtual resultCode run() = 0;
        graph *get_graph() { return g; }
        void set_work_graph(graph *op) { g = op; }
        // addRewriter will create a rewriter using the arguments;
        template <typename T, typename... ARGS>
        void add_rewriter(ARGS... arg)
        {
            auto ptr = std::make_unique<T>(arg...);
            rewriters.push_back(std::move(ptr));
        }

        resultCode apply_rewriter_once(painter &p, graph *g);
        resultCode apply_rewriter_greedy(painter &p, graph *g);

        resultCode walk_apply_rewriter_once(painter &p, graph *g, bool deepwalk = 0);

        // Translation is a special method to apply rewriters,
        // It walk only once through a graph in the dependency order
        // and apply all the applicable rewriters to the ops.
        // So this method is only safe for the case that all rewriters
        // are order free.
        bool translation(painter &p, graph *g);

    public:
        std::vector<std::unique_ptr<rewriterBase>> rewriters;
        std::string _pass_name;
        bool rewriteHappen = 0;
        graph *g = nullptr;
    };

    class passManager
    {
    public:
        passManager() = default;
        passManager(graph *op) { reg = op; }
        void set_work_region(graph *op) { reg = op; }
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
        void init(graph *op) { reg = op; }
        void validation(graph *g);

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

        void add_pass(std::unique_ptr<passBase> ps, graph *g = nullptr)
        {
            if (!g)
                g = reg;
            ps->set_work_graph(dynamic_cast<graph *>(g));
            passes.push_back(std::move(ps));
        }
        void add_normalization_pass();
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
        graph *reg = nullptr;
        std::string name = "";
    };

}

#endif