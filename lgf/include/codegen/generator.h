#ifndef LGF_CODEGEN_GENERATOR_H
#define LGF_CODEGEN_GENERATOR_H
#include "ast/ast.h"
#include "ast/generator.h"
#include "ast/lexer.h"
#include <map>
#include <memory>
#include "ast/stream.h"
using namespace lgf::ast;
namespace lgf::codegen
{
    class generatorBase : public ast::parser
    {
    public:
        generatorBase(const std::string n) : name(n) {}
        virtual ~generatorBase() {}
        virtual std::unique_ptr<astNode> parse(fiostream &fs) = 0;

    private:
        std::string name;
    };

    class generatorHandle
    {
    public:
        generatorHandle() = default;
        ~generatorHandle() = default;
        virtual void init() = 0;
        generatorBase *get()
        {
            if (!is_init)
            {
                init();
            }
            return gen.get();
        }

    private:
        bool is_init = false;
        std::unique_ptr<generatorBase> gen;
    };

    class generatorMap
    {
    public:
        generatorMap() = default;
        ~generatorMap() = default;
        generatorHandle *get(const std::string &name) const
        {
            auto it = gmap.find(name);
            if (it != gmap.end())
            {
                return it->second.get();
            }
            return nullptr;
        }

    private:
        std::map<std::string, std::unique_ptr<generatorHandle>> gmap;
    };
}

#endif