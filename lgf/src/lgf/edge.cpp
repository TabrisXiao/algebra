
#include "lgf/edge.h"
#include "lgf/node.h"

namespace lgf
{
    void edge::decouple()
    {
        if (bundle)
        {
            bundle->need_clean();
        }
        auto target = dual;
        dual = nullptr;
        if (target)
            target->decouple();
    }

    void edgeBundle::clean()
    {
        if (!bNeedClean)
            return;
        auto iter = begin();
        while (iter != end())
        {
            if ((*iter).is_coupled())
            {
                iter++;
            }
            else
            {
                iter = erase(iter);
            }
        }
        bNeedClean = 0;
    }

    void edgeBundle::push_back(edge &&e)
    {
        std::vector<edge>::push_back(std::move(e));
        back().update_bundle(this);
    }
} // namespace lgf