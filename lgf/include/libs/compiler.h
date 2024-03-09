

#ifndef LIBS_COMIPLER_H_
#define LIBS_COMIPLER_H_

#include "lgf/LGFCompiler.h"
#include "libs/aab/AAB.h"
#include "libs/transform/convertToSIO.h"

namespace lgf {
class AABCompiler : public compilerPrototype {
public: 
    AABCompiler() = default;
    void build_pipeline() override {
        pm.addNormalizationPass();
        pm.addPass(AAB::createAAProcess());
        pm.addNormalizationPass();
        pm.addPass(transform::createConvertToSIOPass());
    }
};
}

#endif