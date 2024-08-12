
#ifndef LGF_CODEGEN_NODEGENERATOR_H
#define LGF_CODEGEN_NODEGENERATOR_H
#include "generator.h"
#include "parser.h"
namespace lgf::codegen
{

    class nodeGenerator : public generatorBase
    {
    public:
        nodeGenerator() : generatorBase("node") {}
        virtual ~nodeGenerator() {}
        virtual std::unique_ptr<astModule> parse(lexer &l)
        {

            auto node = std::make_unique<astModule>();
            return std::move(node);
        }
    };
} // namespace lgf::codegen
#endif