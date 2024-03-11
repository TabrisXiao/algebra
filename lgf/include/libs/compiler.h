

#ifndef LIBS_COMIPLER_H_
#define LIBS_COMIPLER_H_

#include "lgf/LGFCompiler.h"
#include "libs/aab/AAB.h"
#include "libs/fa/passes.h"
#include "libs/transform/convertToSIO.h"

namespace lgf {
class AABCompiler : public compilerPrototype {
public: 
    AABCompiler() {};
    void build_pipeline() override {
        pm.addNormalizationPass();
        pm.addPass(AAB::createAAProcess());
        pm.addPass(createCalculusPass());
        pm.addNormalizationPass();
        pm.addPass(transform::createConvertToSIOPass());
    }
};
}

#endif