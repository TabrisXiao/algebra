
#ifndef LGFCONTEXT_H_
#define LGFCONTEXT_H_

#include "type.h"
#include "symbolicTable.h"
#include <memory>
#include <vector>

namespace lgf{

class LGFContext {
    public: 
    using mTable = symbolTable<std::unique_ptr<LGFModule>>;
    using regionTable = nestedSymbolicTable<mTable>;
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
    template<typename module_t>
    module_t* registerModule(std::string name){
        // the name of module should be like
        // region::subregion::module_name
        liteParser parser.loadBuffer(name);
        std::string id = parser.parseIdentifier();
        auto* rptr = root_region.createOrGetTable(id);
        if(parser.getCurToken()==int(':')) 
            parser.parseColon();
        while(parser.getCurToken()==int(':')){
            parser.parseColon();
            id = parser.parseIdentifier();
            if(parser.getCurToken()!=int(':')) break;
            rptr=rptr->createOrGetTable(id);
            parser.parserColon();
        }
        if(auto module = rptr->getData()->find(id)) return module.get();
        return rptr->getData()->addEntry(id, module_t());
    }

    std::vector<std::unique_ptr<typeImpl>> types;
    regionTable root_region;
};

}

#endif