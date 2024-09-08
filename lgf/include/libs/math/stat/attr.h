#ifndef LGF_STAT_ATTR_H
#define LGF_STAT_ATTR_H

#include "libs/builtin/attr.h"

namespace lgf::stat
{
    enum pdfType : uint32_t
    {
        uniform,
        normal,
        exponential,
        poisson,
        binomial,
        chi2,
    };

    class pdfTypeAttr : public enumAttr<pdfType>
    {
    public:
        pdfTypeAttr() : enumAttr<pdfType>("pdfType") {}
        pdfTypeAttr(pdfType t) : enumAttr<pdfType>("pdfType", t) {}
        ~pdfTypeAttr() = default;
        inline static pdfTypeAttr get(pdfType t)
        {
            return pdfTypeAttr(t);
        }
        virtual sid_t represent() override{
            
        }
    };
}

#endif