
#ifndef LGF_LIB_ALGEBRA_DESC_H
#define LGF_LIB_ALGEBRA_DESC_H

#include "lgf/value.h"
#include "lgf/attribute.h"

namespace lgf{

class algebraDesc {
    public:
    algebraDesc() = default;
    
};

class realNumber : public simpleValue
{
    public:
    realNumber() : simpleValue("realNumber") {}
};

class unitDesc : public valueDesc{
    public:
    unitDesc(valueDesc* elem) : valueDesc("unit"), elemDesc(elem) {}
    virtual sid_t represent() override{
        return "unit<"+elemDesc->represent()+">";  
    }
    valueDesc* get_elem_desc(){
        return elemDesc;
    }
    private:
    valueDesc* elemDesc = nullptr;
};

} // namespace lgf

#endif