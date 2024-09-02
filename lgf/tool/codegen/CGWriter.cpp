
#include "CGWriter.h"

void codegen::CGWriter::write_context(astContext *ptr)
{
    auto name = ptr->get_name();
    os.indent() << "namespace " << name << "\n";
    os.indent() << "{\n";
    os.indent_level_up();
    write_content(ptr->get<astList>("_content_"));
    os.decr_indent()
        << "} // namespace " << name << "\n";
}

void codegen::CGWriter::write_content(astList *ptr)
{
    for (auto &it : ptr->get_content())
    {
        auto kind = it->get_kind();
        switch (kind)
        {
        case astNode::context:
            write_context(it->as<astContext>());
            break;
        case astNode::module:
            write_module((it->as<astModule>()));
            break;
        case astNode::dict:
            write_dict((it->as<astDictionary>()));
            break;
        case astNode::list:
            write_list((it->as<astList>()));
            break;
        default:
            break;
        }
    }
}

void codegen::CGWriter::write_module(astModule *ptr)
{
    nodeTemplate a(ptr);
    a.write(os);
    //
}

void codegen::CGWriter::write_op_builder(astModule *ptr)
{
    auto input = ptr->get<astDictionary>("input");
    auto output = ptr->get<astExpr>("output");
    auto name = ptr->get<astExpr>("ir_name");
}
void codegen::CGWriter::write_dict(astDictionary *ptr)
{
}
void codegen::CGWriter::write_list(astList *ptr)
{
}