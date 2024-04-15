
#include "lgf/edge.h"

namespace lgf
{
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