
#ifndef LGF_MODULETABLE_H
#define LGF_MODULETABLE_H
#include "lgf/symbolicTable.h"
#include "lgf/exception.h"
#include <functional>
#include "Builtin/Builtin.h"
#include "AAB/aab.h"
#include "LinearAlg/LinearAlg.h"
#include "lgf/LGFContext.h"

namespace lgf {
class moduleManager : public passManager{
    public:
    using create_pass_func_t = std::function<std::unique_ptr<passBase>(moduleOp*)>;
    using type_reg_func_t = std::function<void(LGFContext*)>;
    struct moduleInfo {
        type_reg_func_t type_reg_f;
        create_pass_func_t init_pass_creator;
    };
    moduleManager(moduleManager &) = delete;
    moduleManager(moduleManager &&) = delete;
    ~moduleManager() { delete _mtble; }
    static moduleManager &get(){
        if (_mtble != nullptr)
            return *_mtble;
        _mtble = new moduleManager();
        return *_mtble;
    }
    moduleInfo* findInfo(std::string id){
        return table.find(id);
    }

    void loadInternalModule(std::string mid, LGFContext *ctx, lgf::moduleOp* module){
        auto entry = _mtble->findInfo(mid);
        THROW_WHEN(entry==nullptr, "The module: "+mid+" is unknown!");
        addPass(entry->init_pass_creator(module));
        entry->type_reg_f(ctx);
    }

    void loadDefaultTranslationPipeline(){
        name = "LGFModule Init (Default Pipeline)";
        addNormalizationPass();
        table.addEntry("Builtin", {&LGFBaseModule::registerTypes, &Builtin::createInterfaceInitPass});
        table.addEntry("LinearAlg", {&LinearAlgModule::registerTypes, &LinearAlg::createInterfaceInitPass});
        table.addEntry("AAB", {&AABModule::registerTypes, &AAB::createInterfaceInitPass});
    }

    protected:
    moduleManager(){}
    
    inline static moduleManager *_mtble=nullptr;
    symbolTable<moduleInfo> table;
};
}
#endif