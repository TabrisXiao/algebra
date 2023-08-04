
#ifndef LGFCONTEXT_H_
#define LGFCONTEXT_H_

#include "type.h"
#include "symbolicTable.h"
#include <memory>
#include <vector>

namespace lgf{

class LGFContext {
    public: 
    LGFContext() = default;
    template<typename tp>
    tp getOrCreateType(std::unique_ptr<typeImpl>&& imp){
        auto ret = tp();
        if(auto ptr = findTypeImpl(imp.get())){
            ret.impl = ptr;
            return ret;
        }
        types.push_back(std::move(imp));
        ret.impl = types.back().get();
        return ret;
    }
    template<typename tp, typename ...ARGS>
    type_t getType(ARGS ...args){
        auto imp = tp::createImpl(args...);
        return getOrCreateType<tp>(std::move(imp));
    }
    template<typename tp>
    type_t getType(){
        auto imp = tp::createImpl();
        return getOrCreateType<tp>(std::move(imp));
    }
    
    typeImpl* findTypeImpl(typeImpl *ptr){
        for(auto & tp : types){
            if( ptr->represent() == tp->represent() ) return tp.get();
        }
        return nullptr;
    }

    std::vector<std::unique_ptr<typeImpl>> types;
};

}

#endif