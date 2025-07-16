#include "operation.h"
#include "env.h"
#include <deque>
namespace lgf
{
    class status
    {
    public:
        status() = default;
        status(int8_t c) : code(c) {}

        static status failure()
        {
            return status(1);
        }
        static status error()
        {
            return status(2);
        }
        bool is_error() const
        {
            return code == 2;
        }
        bool is_fail() const
        {
            return code == 1;
        }
        static status success()
        {
            return status(1);
        }
        bool is_success()
        {
            return code == 0;
        }
        int8_t code = 0;
    };

    class patternWriterBase
    {
    public:
        patternWriterBase() = default;
        virtual status run(operation *) = 0;
    };

    template <typename T>
    class opWriter : public patternWriterBase
    {
    public:
        virtual status run(operation *op) override
        {
            if (auto t = dynamic_cast<T *>(op))
            {
                return rewrite(t);
            }
            return status::failure();
        }
        virtual status rewrite(T *op) = 0;
    };

    class processBase
    {
    public:
        processBase(const char *pname) : name(pname) {};
        virtual ~processBase() = default;
        virtual status run(region *r) = 0;
        region *get_region()
        {
            return g_;
        }
        void set_region(region *r)
        {
            g_ = r;
        }

        status applyRewriteOnce()
        {
            auto res = status::failure();
            for (auto &p : patterns)
            {
                for (auto &n : g_->nodes)
                {
                    auto op = dynamic_cast<operation *>(n.get());
                    if (!op->is_valid())
                        continue;
                    if (p->run(op).is_success())
                    {
                        res = status::success();
                    }
                }
            }
            g_->clean();
            return res;
        }
        const std::string &get_name() const
        {
            return name;
        }
        status applyRewriteGreedy()
        {
            auto res = status::failure();
            auto flag = applyRewriteOnce();
            while (flag.is_success())
            {
                flag = applyRewriteOnce();
                res = status::success();
            }
            return res;
        }
        std::vector<std::unique_ptr<patternWriterBase>> patterns;
        region *g_ = nullptr; // the region this transform applies to
        std::string name;     // name of the process, used for logging
    };

    class processManager
    {
    public:
        processManager(region *r) : reg_(r) {}
        void run()
        {
            for (auto &todo : todos)
            {
                if (todo->run(reg_).is_error())
                {
                    THROW("Failed at process: " + todo->get_name());
                }
            }
        }
        std::deque<std::unique_ptr<processBase>> todos;
        region *reg_ = nullptr;
    };
}