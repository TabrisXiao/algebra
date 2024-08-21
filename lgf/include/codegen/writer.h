
#ifndef LGF_CODEGEN_WRITER_H
#define LGF_CODEGEN_WRITER_H
#include <map>
#include "utils/stream.h"
#include "ast/ast.h"
#include "lgf/utils.h"
using namespace utils;
namespace lgf::codegen
{
    class writer
    {
    public:
        writer() = default;
        virtual ~writer() = default;
        virtual logicResult run(ast::astModule *) = 0;
        logicResult write(ast::astModule *r, cgstream &fs)
        {
            cg = &fs;
            return run(r);
        }
        std::string chain_string(std::vector<std::string> &v)
        {
            std::string str = "";
            for (auto &s : v)
            {
                str = str + s + ", ";
            }
            return str.substr(0, str.size() - 2);
        }

        cgstream &os() { return *cg; }

    private:
        cgstream *cg;
    };

    class nodeWriter : public writer
    {
    public:
        nodeWriter() = default;
        virtual ~nodeWriter() = default;
        virtual logicResult run(ast::astModule *r) override
        {
            root = r;
            root_attr = root->get_attr();
            auto parents = root_attr->get<ast::astList>("_inherit_");
            std::string parent_str = "";
            for (auto &ptr : parents->get_content())
            {
                auto parent = dynamic_cast<ast::astExpr *>(ptr.get())->get_expr();
                parent_str = parent_str + "public " + parent + ", ";
            }
            parent_str = parent_str.substr(0, parent_str.size() - 2);
            if (parent_str.size() > 0)
                parent_str = ": " + parent_str;
            auto name = root->get_name();
            os() << "class " << name << parent_str + " {\n";
            os().incr_indent_level();
            os() << "public:\n";
            write_property(name);
            write_build_func(name);
            os().decr_indent_level();
            os() << "};\n";
            return logicResult::success();
        }
        void write_property(const std::string &n)
        {
            auto alias = n;
            if (root_attr->find("ir_name").is_success())
            {
                alias = root_attr->get<ast::astExpr>("ir_name")->get_expr();
            }
            os().indent() << n << "(): node(\"" << alias << "\")";

            if (root_attr->find("property").is_success())
            {
                os() << "\n";
                os().indent() << "{\n";
                os().incr_indent_level();
                auto propInfo = root_attr->get<ast::astList>("property");
                for (auto &ptr : propInfo->get_content())
                {
                    auto prop = dynamic_cast<ast::astExpr *>(ptr.get())->get_expr();
                    if (prop == "trivial")
                    {
                        os().indent() << "mark_status(eIdenticalRemovable);\n";
                    }
                    else
                    {
                        std::cerr << "Unknown property: " << prop << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                os().decr_indent_level();
                os().indent() << "}\n\n";
            }
            else
            {
                os() << " {}\n\n";
            }

            os().indent() << "virtual ~" << n << "() = default;\n\n";
        }

        void write_build_func(std::string name)
        {
            std::string arg_str = "";
            if (root_attr->find("input").is_success())
            {
                auto inputs = root_attr->get<ast::astDictionary>("input");
                auto &contents = inputs->get_contents();
                for (auto &c : contents)
                {
                    auto key = c.first;
                    auto value = c.second.get();
                    auto type = dynamic_cast<ast::astExpr *>(value)->get_expr();
                    arg_type.push_back(type);
                    arg_str = arg_str + "node*" + " " + key + ", ";
                    arg_id.push_back(key);
                }

                arg_str = arg_str.substr(0, arg_str.size() - 2);
            }
            if (root_attr->find("output").is_success())
            {
                auto retop = root_attr->get<ast::astDictionary>("output");
                for (auto &it : retop->get_contents())
                {
                    output_id = it.first;
                    output_type = dynamic_cast<ast::astExpr *>(it.second.get())->get_expr();
                    arg_str = "descriptor " + output_id + ", " + arg_str;
                }
            }
            if (arg_str.size() > 0)
                arg_str = ", " + arg_str;
            os().indent() << "static " << name << "* build" << "(LGFContext* ctx" << arg_str << ") {\n";
            os().incr_indent() << "auto op = new " << name << "();\n";
            auto arg_list = chain_string(arg_id);
            os().indent() << "op->register_input(" + arg_list + ");\n";
            if (output_id.size() > 0)
            {
                os().indent() << "op->set_value_desc(" << output_id << ");\n";
            }
            os().indent() << "return n;\n";
            os().decr_indent() << "}\n\n";
            for (auto i = 0; i < arg_id.size(); i++)
            {
                os().indent() << "node* " << arg_id[i] << "(){\n";
                os().incr_indent() << "return input(" << i << ");\n";
                os().decr_indent() << "}\n\n";
            }
        }
        ast::astModule *root;
        ast::astDictionary *root_attr;
        std::vector<std::string> arg_type, arg_id;
        std::string output_id, output_type;
    };

    class writerManager
    {
    public:
        writerManager()
        {
            wmap["node"] = std::make_unique<nodeWriter>();
        }
        std::string get_first_attr(ast::astModule *node)
        {
            auto attr = node->get_attr();
            return attr->get<ast::astList>("_attr_")->get<ast::astExpr>(0)->get_expr();
        }
        virtual ~writerManager() = default;

        writer *get_writer(const std::string &sid)
        {
            if (wmap.find(sid) == wmap.end())
            {
                return nullptr;
            }
            return wmap[sid].get();
        }
        logicResult process(std::map<std::string, std::unique_ptr<ast::astModule>> &m, cgstream &fs)
        {
            for (auto &item : m)
            {
                auto sm = dynamic_cast<ast::astModule *>(item.second.get());
                auto w = get_writer(get_first_attr(sm));
                if (w == nullptr || w->write(sm, fs).is_fail())
                    return logicResult::fail();
            }
            return logicResult::success();
        }

    private:
        std::map<std::string, std::unique_ptr<writer>> wmap;
    };

} // namespace lgf::codegen

#endif