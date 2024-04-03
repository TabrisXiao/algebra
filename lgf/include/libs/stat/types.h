#ifndef LGF_STAT_TYPES_H_
#define LGF_STAT_TYPES_H_

#include "libs/Builtin/types.h"
#include "lgf/LGFContext.h"
#include "libs/aab/aab.h"
#include "lgf/utils.h"

namespace lgf{
    
class randomVariable : public variable {
    public:
    using desc_t = descriptor;
    static inline const sid_t sid= "randomVariable";
    randomVariable(){ }
};

class normalDesc : public descriptor, public algebraAxiom {
    public:
    normalDesc(double m = 0, double v = 1): mu(m), sigma(v){
        initAsField();
    }
    virtual std::string representType()const override{
        std::string str = getSID()+"(mean="+lgf::to_string(mu)+",variance="+lgf::to_string(sigma)+")";
        return str;
    }
    double mean(){ return mu; }
    double variance(){ return sigma; }
    protected:
    double mu, sigma;
};

class normalVariable : public randomVariable {
    public:
    using desc_t = normalDesc;
    static inline const sid_t sid= "stat::Normal";
    normalVariable() = default;
};

}// namespace lgf

#endif