
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
    class templateBase
    {
    public:
        templateBase(CGContext *c, astModule *m, dependencyMap *dp)
        {
            dep = dp;
            module = m;
            ctx = c;
            name = m->get_name();
        }
        std::string str(astNode *ptr)
        {
            return dynamic_cast<astExpr *>(ptr)->string();
        }
        virtual void write(stringBufferProducer &os) = 0;
        std::string name;
        CGContext *ctx;
        astModule *module;
        dependencyMap *dep;
        astList *parents;
    };

    class attrTemplate : public templateBase
    {
    public:
        attrTemplate(CGContext *ctx, astModule *m, dependencyMap *dp) : templateBase(ctx, m, dp)
        {
            parents = m->get<astList>("_parent_");
            for (auto &it : parents->get_content())
            {
                inheritStr = inheritStr + ", public ::lgf::" + str(it.get());
                dp->include(str(it.get()).c_str());
            }
        }
        virtual void write(stringBufferProducer &os) final
        {
            os.indent() << "class " << name << " : " << inheritStr << "\n";
            os.indent() << "{\n";
            os.indent() << "public:\n";
            os.incr_indent() << name << "() = default;\n";
            os.decr_indent() << "};\n";
        }

        std::string inheritStr = "public ::lgf::attribute";
    };

    class nodeTemplate : public templateBase
    {
    public:
        class argID
        {
            std::string name, type;
        };
        enum property_kind
        {
            p_region = 1,
        };
        nodeTemplate(CGContext *ctx, astModule *m, dependencyMap *dp) : templateBase(ctx, m, dp)
        {
            parents = m->get<astList>("_parent_");
            args = m->get<astDictionary>("input");
            output = m->get<astExpr>("output");
            alias = m->get<astExpr>("ir_name");
            extra = m->get<astExpr>("_extra_");
            auto proNode = m->get<astList>("property");
            if (proNode)
            {
                for (auto &it : proNode->get_content())
                {
                    auto str = it->as<astExpr>()->string();
                    if (str == "region")
                    {
                        properties.push_back(p_region);
                    }
                }
            }

            for (auto &it : parents->get_content())
            {
                inheritStr = inheritStr + ", public ::lgf::" + str(it.get());
                dp->include(str(it.get()).c_str());
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
                auto info = ctx->get_info(argType);
                if (!info)
                {
                    arg.second->emit_error("Type not found: " + argType);
                }
                std::string type = "node *";
                if (info->stype == symbolInfo::attr)
                {
                    type = "attribute ";
                    argAttr.push_back(argName);
                }
                else if (info->stype == symbolInfo::desc)
                {
                    argChain += argName + ", ";
                    argDesc.push_back(argName);
                }
                else
                {
                    arg.second->emit_error("Invalid type for input: " + argType);
                }
                inputArgStr = inputArgStr + ", " + type + argName;
            }
        }
        std::string str(astNode *ptr)
        {
            return dynamic_cast<astExpr *>(ptr)->string();
        }
        virtual void write(stringBufferProducer &os) final
        {
            os.indent() << "class " << name << " : " << inheritStr << "\n";
            os.indent() << "{\n";
            os.indent() << "private:\n";
            os.incr_indent() << name << "() = default;\n\n";
            os.decr_indent() << "public:\n";
            // create build function
            os.incr_indent() << "static " << name << " *build(::lgf::LGFContext *ctx" << inputArgStr << outputArgStr << ") \n";
            os.indent() << "{\n";
            os.incr_indent() << "auto op = new " << name << "();\n";
            if (argChain.size() > 0)
            {
                argChain = argChain.substr(0, argChain.size() - 2);
                os.indent() << "op->register_input(" << argChain << ");\n";
            }
            if (argAttr.size() > 0)
            {
                for (auto &it : argAttr)
                {
                    os.indent() << "op->add_attr(\"" << it << "\", " << it << ");\n";
                }
            }
            if (output)
                os.indent() << "op->set_value_desc(output_type);\n";
            if (alias)
            {
                os.indent() << "op->set_sid(\"" << alias->string() << "\");\n";
            }
            if (has_property(p_region))
            {
                os.indent() << "op->create_region();\n";
            }
            os.indent() << "return op;\n";
            os.decr_indent() << "}\n";
            // -----------
            // arg function:
            if (argDesc.size() > 0)
            {
                int i = 0;
                for (auto &it : argDesc)
                {
                    os.indent() << "node *" << it << "() { return input(" << std::to_string(i++) << "); }\n";
                }
            }
            if (argAttr.size() > 0)
            {
                for (auto &it : argAttr)
                {
                    os.indent() << "attribute " << it << "() { return get_attr(\"" << it << "\"); }\n";
                }
            }

            // extra definition goes here
            if (extra)
            {
                os << extra->string() << "\n";
            }

            os.decr_indent() << "};\n";
        }
        bool has_property(property_kind kind)
        {
            return std::find(properties.begin(), properties.end(), kind) != properties.end();
        }

        std::string inheritStr = "public ::lgf::node";
        std::string outputArgStr, inputArgStr;
        astExpr *alias;
        std::vector<std::string> argDesc;
        std::vector<std::string> argAttr;
        astDictionary *args;
        std::string argChain = "";
        astExpr *output = nullptr;
        astList *parents;
        astExpr *extra = nullptr;
        std::vector<property_kind> properties;
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
            ctx->reset();
            depTable = dep;
            os.clear();
            osHeader.clear();
            dep->flush();
            auto content = r->get<astList>("_content_");
            std::string uid;
            for (auto &ptr : content->get_content())
            {
                if (ptr->get_kind() != astNode::expr)
                    continue;
                uid = ptr->as<astExpr>()->string();
            }
            if (uid.size() == 0)
                std::runtime_error("UID guard not found!");
            write_content(content);
            os << "#endif // " << uid << "\n";
            std::vector<stringBuf> ret;
            ret.push_back(get_headers(r, uid));
            ret.push_back(os.write_to_buffer());
            return ret;
        }

        std::string get_str_attr(astDictionary *ptr, const std::string &key)
        {
            return ptr->get<astExpr>(key)->string();
        }
        stringBuf get_headers(ast::astContext *root, std::string uid)
        {
            std::string str = "#ifndef " + uid + "\n#define " + uid + "\n";
            auto import = root->get<astList>("_import_");
            if (import)
            {
                for (auto &it : import->get_content())
                {
                    str = str + "#include \"" + it->as<astExpr>()->string() + "\"\n";
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