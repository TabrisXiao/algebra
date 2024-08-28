
#include "CGWriter.h"

void codegen::CGWriter::write_context(astContext *ptr, cgstream &os)
{
    auto name = ptr->get_name();
    os.indent() << "namespace " << name << " {\n";
    os.incr_indent_level();
    for (auto &it : ptr->get_contents())
    {
        auto kind = it.second->get_kind();
        switch (kind)
        {
        case astNode::context:
            write_context(it.second->as<astContext>(), os);
            break;
        case astNode::module:
            write_module((it.second.get()->as<astModule>()), os);
            break;
        case astNode::dict:
            write_dict((it.second.get()->as<astDictionary>()), os);
            break;
        case astNode::list:
            write_list((it.second.get()->as<astList>()), os);
            break;
        default:
            break;
        }
    }
    os.decr_indent() << "}\n // namespace " << name << "\n\n";
}
