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
    root->add("_content_", std::move(std::make_unique<astList>(parser.loc())));
    parse_file(ctx, &parser, root.get());
    includes.pop_back();
    return std::move(root);
}

void codegen::CGCompiler::parse_file(CGContext *ctx, CGParser *p, astContext *root)
{
    p->reset();
    auto content = root->get<astList>("_content_");
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
            parse_import(ctx, p);
        }
        else
        {
            p->emit_error("Unknown key word: " + id);
        }
    }
}

void codegen::CGCompiler::parse_import(CGContext *ctx, CGParser *parser)
{
    std::filesystem::path p = ".";
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
    newlex.load_stringBuf(p.string(), buffer);
    include(fp.parent_path());
    CGParser np(newlex);

    astContext temp(parser->loc());
    temp.add("_content_", std::move(std::make_unique<astList>(parser->loc())));

    parse_file(ctx, &np, &temp);
    includes.pop_back();
}