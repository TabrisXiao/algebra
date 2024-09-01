
#ifndef LGF_CODEGEN_WRITER_H
#define LGF_CODEGEN_WRITER_H
#include <map>
#include "aoc/stream.h"
#include "aoc/convention.h"
#include "ast/ast.h"
#include "CGContext.h"

using namespace ast;
using namespace aoc;
namespace codegen
{

    class nodeTemplate
    {
    public:
        nodeTemplate(astModule *m)
        {
            name = m->get_name();
            parents = m->get<astList>("_parent_");
            args = m->get<astDictionary>("input");
            output = m->get<astExpr>("output");
            alias = m->get<astExpr>("ir_name");
            for (auto &it : parents->get_content())
            {
                inheritStr = inheritStr + "public " + str(it.get()) + ", ";
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
        stringBuf write(CGContext *c, ast::astContext *r)
        {
            ctx = c;
            os.clear();
            write_context(r);
            return os.write_to_buffer();
        }

        std::string get_str_attr(astDictionary *ptr, const std::string &key)
        {
            return ptr->get<astExpr>(key)->string();
        }
        void write_op_builder(astModule *ptr);
        void write_context(astContext *ptr);
        void write_module(astModule *ptr);
        void write_dict(astDictionary *ptr);
        void write_list(astList *ptr);

    private:
        stringBufferProducer os;
        CGContext *ctx;
    };

} // namespace codegen

#endif