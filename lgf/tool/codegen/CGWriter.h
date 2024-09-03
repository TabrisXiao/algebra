
#ifndef LGF_CODEGEN_WRITER_H
#define LGF_CODEGEN_WRITER_H
#include <map>
#include "aoc/stream.h"
#include "aoc/convention.h"
#include "ast/ast.h"
#include "CGContext.h"
#include "dependency.h"

using namespace ast;
using namespace aoc;
namespace codegen
{

    class nodeTemplate
    {
    public:
        nodeTemplate(astModule *m, dependencyMap *dp)
        {
            name = m->get_name();
            parents = m->get<astList>("_parent_");
            args = m->get<astDictionary>("input");
            output = m->get<astExpr>("output");
            alias = m->get<astExpr>("ir_name");
            for (auto &it : parents->get_content())
            {
                inheritStr = inheritStr + "public " + str(it.get()) + ", ";
                dp->include(str(it.get()).c_str());
            }
            if (inheritStr.size() == 0)
            {
                inheritStr = "public node";
            }
            else
            {
                inheritStr = inheritStr.substr(0, inheritStr.size() - 2);
            }

            if (output)
            {
                outputArgStr = ", descriptor output_type";
            }
            if (!args)
                return;
            for (auto &arg : args->get_content())
            {
                auto argName = arg.first;
                auto argType = str(arg.second.get());
                inputArgStr = inputArgStr + ", node *" + argName;
            }
        }
        std::string str(astNode *ptr)
        {
            return dynamic_cast<astExpr *>(ptr)->string();
        }
        std::string make_arg_chain()
        {
            std::string argStr = "";
            for (auto &arg : args->get_content())
            {
                argStr = argStr + arg.first + ", ";
            }
            return argStr.substr(0, argStr.size() - 2);
        }
        void write(stringBufferProducer &os)
        {
            os.indent() << "class " << name << " : " << inheritStr << "\n";
            os.indent() << "{\n";
            os.indent() << "private:\n";
            os.incr_indent() << name << "() = default;\n\n";
            os.decr_indent() << "public:\n";
            // create build function
            os.incr_indent() << "static " << name << " *build(LGFContext *ctx" << inputArgStr << outputArgStr << ") \n";
            os.indent() << "{\n";
            os.incr_indent() << "auto op = new " << name << "();\n";
            if (args)
            {
                os.indent() << "op->register_input(" << make_arg_chain() << ");\n";
            }
            if (output)
                os.indent() << "op->set_value_desc(output_type);\n";
            if (alias)
            {
                os.indent() << "op->set_sid(\"" << alias->string() << "\");\n";
            }
            os.indent() << "return op;\n";
            os.decr_indent() << "}\n";
            // -----------

            os.decr_indent() << "};\n";
        }

        std::string inheritStr = "";
        std::string outputArgStr, inputArgStr;
        std::string name;
        astExpr *alias;
        astDictionary *args;
        astExpr *output = nullptr;
        astList *parents;
    };
    class CGWriter
    {
    public:
        CGWriter() = default;
        virtual ~CGWriter() = default;
        std::vector<stringBuf> convert(CGContext *c, ast::astContext *r, dependencyMap *dep, std::string base)
        {
            baseStr = base;
            depTable = dep;
            ctx = c;
            depTable = dep;
            os.clear();
            osHeader.clear();
            dep->flush();
            auto content = r->get<astList>("_content_");
            write_content(content);

            std::vector<stringBuf> ret;
            ret.push_back(get_headers(r));
            ret.push_back(os.write_to_buffer());
            return ret;
        }

        std::string get_str_attr(astDictionary *ptr, const std::string &key)
        {
            return ptr->get<astExpr>(key)->string();
        }
        stringBuf get_headers(ast::astContext *root)
        {
            std::string str = "";
            auto import = root->get<astList>("_import_");
            if (import)
            {
                for (auto &it : import->get_content())
                {
                    str = str + "#include \"" + baseStr + "/" + it->as<astExpr>()->string() + "\"\n";
                }
                std::replace(str.begin(), str.end(), '\\', '/');
            }
            for (auto &it : depTable->get())
            {
                str = str + "#include \"" + it + "\"\n";
            }
            depTable->flush();
            return stringBuf(std::move(str));
        }
        void write_content(astList *ptr);
        void write_op_builder(astModule *ptr);
        void write_context(astContext *ptr);
        void write_module(astModule *ptr);
        void write_dict(astDictionary *ptr);
        void write_list(astList *ptr);

    private:
        stringBufferProducer os, osHeader;
        CGContext *ctx;
        dependencyMap *depTable;
        std::string baseStr;
    };

} // namespace codegen

#endif