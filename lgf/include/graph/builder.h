#ifndef LGF_PAINTER_H_
#define LGF_PAINTER_H_
#include "env.h"
#include "object.h"
#include "operation.h"
#include "context.h"
#include <deque>

namespace lgf
{
    class builder
    {
    public:
        using wptr_t = std::deque<std::unique_ptr<node>>::iterator;
        builder(context *c) : rg(c->get_region()) {}
        template <typename T>
        T *create()
        {
            if (rg == nullptr)
            {
                throw std::runtime_error("work region is not set for the builder.");
            }
            auto op = T::build(rg);
            auto res = op.get();
            wptr = rg->nodes.insert(wptr, std::move(op)) + 1;
            return res;
        }
        template <typename T, typename... ARGS>
        T *create(ARGS... args)
        {
            if (rg == nullptr)
            {
                throw std::runtime_error("work region is not set for the builder.");
            }
            auto op = T::build(rg, args...);
            auto res = op.get();
            wptr = rg->nodes.insert(wptr, std::move(op)) + 1;
            return res;
        }

        void move_to_region_start()
        {
            wptr = rg->nodes.begin();
        }

        void goto_region(region *r)
        {
            rg = r;
            wptr = rg->nodes.begin();
        }

        wptr_t wptr; // pointer to the insertion position in the graph

        region *rg = nullptr;
    };

} // namespace lgf
#endif