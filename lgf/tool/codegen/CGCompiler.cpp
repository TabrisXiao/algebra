#include "CGCompiler.h"
using kind_t = ast::token::kind;
std::unique_ptr<astContext> codegen::CGCompiler::parse(CGContext *ctx, sfs::path input)
{
    auto buffer = load_file_to_string_buffer(input);
    lexer lx;
    lx.load_stringBuf(input.string(), buffer);
    CGParser parser(lx);
    include(input.parent_path());
    auto root = std::make_unique<astContext>(parser.loc());
    root->set_name("_root_");
    root->add("_import_", std::move(std::make_unique<astList>(parser.loc())));
    root->add("_content_", std::move(std::make_unique<astList>(parser.loc())));
    parse_file(ctx, &parser, root.get(), inputBase);
    includes.pop_back();
    return std::move(root);
}

void codegen::CGCompiler::parse_file(CGContext *ctx, CGParser *p, astContext *root, sfs::path baseFolder)
{
    p->reset();
    CGContext::CGCGuard guard(ctx);
    auto id = p->parse_id();
    if (id != "uid")
    {
        p->emit_error("Expecting UID at beginning of file");
    }
    auto content = root->get<astList>("_content_");
    content->add(std::make_unique<astExpr>(p->loc(), p->parse_id()));
    while (!p->cur_tok().is(kind_t::tok_eof))
    {
        auto id = p->parse_id();
        if (id == "context")
        {
            content->add(std::move(p->parse_context(ctx)));
        }
        else if (id == "module")
        {
            auto module = p->parse_module(ctx);
            auto name = module->get_name();
            content->add(std::move(module));
        }
        else if (id == "import")
        {
            auto importNode = root->get<astList>("_import_");
            auto path = parse_import(ctx, p, inputBase);
            auto rel = path.lexically_relative(inputBase).replace_extension(".h");
            importNode->add(std::make_unique<astExpr>(p->loc(), rel.string()));
        }
        else
        {
            p->emit_error("Unknown key word: " + id);
        }
    }
}

sfs::path codegen::CGCompiler::parse_import(CGContext *ctx, CGParser *parser, sfs::path baseFolder)
{
    std::filesystem::path p = "";
    do
    {
        auto id = parser->parse_id();
        p /= id;
    } while (parser->try_consume(kind_t('.')).is_success());
    p.replace_extension(".cg");
    auto fp = search_file(p);
    if (fp.empty())
    {
        parser->emit_error("Import file not found: " + p.string());
    }
    lexer newlex;
    auto buffer = load_file_to_string_buffer(fp.string());
    newlex.load_stringBuf(fp.string(), buffer);
    include(fp.parent_path());
    CGParser np(newlex);

    astContext temp(parser->loc());
    temp.add("_content_", std::move(std::make_unique<astList>(parser->loc())));
    parse_file(ctx, &np, &temp, baseFolder);
    includes.pop_back();
    p.replace_extension(".h");
    return fp;
}