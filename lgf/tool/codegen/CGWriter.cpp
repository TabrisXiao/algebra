
#include "CGWriter.h"

void codegen::CGWriter::write_context(astContext *ptr)
{
    auto name = ptr->get_name();
    os.indent() << "namespace " << name << " {\n";
    os.indent_level()++;
    for (auto &it : ptr->get<astDictionary>("_content_")->get_contents())
    {
        auto kind = it.second->get_kind();
        switch (kind)
        {
        case astNode::context:
            write_context(it.second->as<astContext>());
            break;
        case astNode::module:
            write_module((it.second.get()->as<astModule>()));
            break;
        case astNode::dict:
            write_dict((it.second.get()->as<astDictionary>()));
            break;
        case astNode::list:
            write_list((it.second.get()->as<astList>()));
            break;
        default:
            break;
        }
    }
    os.decr_indent() << "}\n // namespace " << name << "\n";
}

void codegen::CGWriter::write_module(astModule *ptr)
{
    auto name = ptr->get_name();
    std::vector<std::unique_ptr<astNode>> *parents = nullptr;
    std::string inheritStr = "";
    if (ptr->has("_parent_"))
    {
        inheritStr = ": ";
        parents = &(ptr->get<astList>("_parent_")->get_content());
        for (auto &p : *parents)
        {
            inheritStr = inheritStr + "public " + p->as<astExpr>()->string() + ", ";
        }
        inheritStr = inheritStr.substr(0, inheritStr.size() - 2);
    }
    os.indent() << "class " << name << inheritStr << " {\n";
    os.indent() << "};\n\n";
    //
}
void codegen::CGWriter::write_dict(astDictionary *ptr)
{
}
void codegen::CGWriter::write_list(astList *ptr)
{
}