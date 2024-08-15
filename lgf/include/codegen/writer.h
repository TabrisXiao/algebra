
#ifndef LGF_CODEGEN_WRITER_H
#define LGF_CODEGEN_WRITER_H
#include <map>
#include "ast/stream.h"
#include "ast/ast.h"
#include "uid.h"
#include "lgf/utils.h"

namespace lgf::codegen
{
    class writer
    {
    public:
        writer() = default;
        virtual ~writer() = default;
        virtual bool run(ast::astDictionary *) = 0;
        bool write(ast::astDictionary *r, ast::cgstream &fs)
        {
            cg = &fs;
            return run(r);
        }

        ast::cgstream &os() { return *cg; }

    private:
        ast::cgstream *cg;
    };

    class nodeWriter : public writer
    {
    public:
        nodeWriter() = default;
        virtual ~nodeWriter() = default;
        virtual bool run(ast::astDictionary *r) override
        {
            root = r;
            auto name = root->get<ast::astExpr>("name")->get_expr();
            os() << "class " << name << ": public node {\n";
            os().incr_indent_level();
            os() << "public:\n";
            os().indent() << name << "(): node(\"" << name << "\") {}\n";
            os().indent() << "virtual ~" << name << "() = default;\n";
            write_build_func(name);
            os().decr_indent_level();
            os() << "};\n";
            return false;
        }

        void write_build_func(std::string name)
        {
            std::string arg_str = "";
            if (root->find("input").is_success())
            {
                auto inputs = root->get<ast::astDictionary>("input");
                auto &contents = inputs->get_contents();
                for (auto &c : contents)
                {
                    auto key = c.first;
                    auto value = c.second.get();
                    auto type = dynamic_cast<ast::astVar *>(value)->get_type_id();
                    arg_str = arg_str + type + " " + key + ", ";
                }

                arg_str = arg_str.substr(0, arg_str.size() - 2);
            }
            if (arg_str.size() > 0)
                arg_str = ", " + arg_str;
            os().indent() << "static " << name << "* build" << "(LGFContext* ctx" << arg_str << ") {\n";
            os().incr_indent()<< "auto n = new " << name << "();\n";
            os().indent() << "return n;\n";
            os().decr_indent_level();
            os().indent() << "}\n";
        }
        ast::astDictionary *root;
    };

    class writerManager
    {
    public:
        writerManager()
        {
            wmap[uid::uid_node] = std::make_unique<nodeWriter>();
        }
        virtual ~writerManager() = default;
        void add_writer(size_t id, std::unique_ptr<writer> w)
        {
            wmap[id] = std::move(w);
        }
        writer *get_writer(size_t id)
        {
            if (wmap.find(id) == wmap.end())
            {
                return nullptr;
            }
            return wmap[id].get();
        }
        bool process(ast::astModule &root, ast::cgstream &fs)
        {
            cg = &fs;
            for (auto &node : root.get_nodes())
            {
                auto sm = dynamic_cast<ast::astDictionary *>(node.get());
                auto uid = sm->get<astNumber>("uid")->get<uid::uid_t>();
                auto w = get_writer(uid);
                if (w != nullptr)
                {
                    if (w->write(sm, *cg))
                        return true;
                }
            }
            return false;
        }

    private:
        std::map<size_t, std::unique_ptr<writer>> wmap;
        ast::cgstream *cg;
    };

} // namespace lgf::codegen

#endif