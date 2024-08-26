
#include "CGParser.h"
using namespace ast;
using kind_t = ast::token::kind;
std::unique_ptr<ast::astDictionary> &&CGParser::parse_context()
{
    // assuming key word 'context' is already parsed and the current char is '<' or a char.
    auto ctx = std::make_unique<ast::astDictionary>(loc());
    add_type("context", ctx.get());
    if (try_consume(kind_t('<')).is_success())
    {
        auto attrs = parse_set();
        ctx->add("_attr_", std::move(attrs));
    }
    auto id = parse_id();
    ctx->add("_name_", std::move(std::make_unique<ast::astExpr>(loc(), id)));
    parse_left_brace();
    auto content = std::make_unique<ast::astDictionary>(loc());
    while (!cur_tok().is_any(kind_t('}'), kind_t::tok_eof))
    {
        auto key = parse_id();
        if (key == "module")
        {
            auto module = parse_module();
            content->add(module->get_name(), std::move(module));
        }
        else if (key == "context")
        {
            auto sub = parse_context();
            auto name = sub->get<ast::astExpr>("_name_")->get_expr();
            content->add(name, std::move(sub));
        }
        else
        {
            emit_error("Unknown key word: " + key.data());
        }
    }
    ctx->add("_content_", std::move(content));
    return std::move(ctx);
}

std::unique_ptr<ast::astDictionary> &&CGParser::parse()
{
    // return true if error
    auto root = std::make_unique<ast::astDictionary>(loc());
    add_type("context", root.get());
    auto id = parse_id();
    if (id != "context")
    {
        THROW("Parse error: 'context' missing at the beginning of " + loc().print());
    }
    return std::move(parse_context());
}

std::unique_ptr<ast::astDictionary> CGParser::parse_dict()
{
    // a dictionary doesn't allow duplicate keys
    // the syntax is {key1: value1, key2: value2, key3: value3...}
    // assuming the beginning of '{' is already parsed
    auto node = std::make_unique<ast::astDictionary>(loc());
    add_type("dict", node.get());
    while (try_consume(kind_t('}')).is_fail())
    {
        auto key = parse_id();
        parse_colon();
        auto &tok = cur_tok();
        switch (tok.get_kind())
        {
        case kind_t('{'):
            node->add(key.data(), std::move(parse_dict()));
            break;
        case kind_t::tok_identifier:
            auto content = parse_id();
            node->add(key.data(), std::move(std::make_unique<ast::astExpr>(loc(), content)));
            break;
        case kind_t('['):
            node->add(key.data(), std::move(parse_list()));
            break;
        case kind_t('<'):
            node->add(key.data(), std::move(parse_set()));
            break;
        default:
            emit_error("Unknown dictionary item!");
        }
        consume();
    }
    return std::move(node);
}

std::unique_ptr<ast::astList> &&CGParser::parse_list()
{
    // a list allows duplicate items
    // the syntax is [item1, item2, item3...]
    auto node = std::make_unique<ast::astList>(loc());
    while (!cur_tok().is(kind_t(',')))
    {
        auto &tok = cur_tok();
        switch (tok.get_kind())
        {
        case kind_t(']'):
            consume();
            return std::move(node);
        case kind_t::tok_identifier:
            node->add(std::move(std::make_unique<ast::astExpr>(loc(), tok.get_string())));
            break;
        case kind_t::tok_integer:
            node->add(std::move(std::make_unique<ast::astExpr>(loc(), tok.get_string())));
            break;
        default:
            emit_error("Unknown list item!");
            break;
        }
        consume();
    }
    return std::move(node);
}

std::unique_ptr<ast::astList> &&CGParser::parse_set()
{
    // set is a list that can't have duplicate items
    // the syntax is <item1, item2, item3...>
    // assuming the beginning of '<' is already parsed
    auto node = std::make_unique<ast::astList>(loc());
    std::set<std::string> key;
    while (!cur_tok().is(kind_t(',')))
    {
        if (cur_tok() == token('>'))
            break;
        if (cur_tok() == token::tok_identifier)
        {
            bool has = 0;
            for (auto &item : node->get_content())
            {
                if (item->get_kind() != ast::astType::expr)
                    continue;
                auto expr = dynamic_cast<ast::astExpr *>(item.get())->get_expr();
                if (expr == get_string())
                {
                    has = 1;
                    THROW("Parse error: Duplicate item: " + expr + " at " + loc().print());
                }
            }
            node->add(std::move(std::make_unique<ast::astExpr>(loc(), get_string())));
        }
        else if (cur_tok() == token::tok_number)
        {
            bool has = 0;
            for (auto &item : node->get_content())
            {
                if (item->get_kind() != ast::astType::number)
                    continue;
                auto num = dynamic_cast<ast::astNumber *>(item.get())->get<double>();
                if (num == get_number())
                {
                    has = 1;
                    THROW("Parse error: Duplicate number: " + std::to_string(num) + " at " + loc().print());
                }
            }
            node->add(std::move(std::make_unique<ast::astNumber>(loc(), get_number())));
        }
        else
        {
            THROW("Parse error: Unknown set item at " + loc().print());
        }
    }
    return std::move(node);
}

std::unique_ptr<ast::astModule> &&codegen::CGParser::parse_module()
{
    // a module is a dictionary
    // the syntax is module<attr1, attr2, attr3...> name : inherit1, inherit2, inherit3...
    // { key1: value1, key2: value2, key3: value3...}
    parse_less_than();
    auto attrs = parse_set();
    auto name = parse_id();
    auto node = std::make_unique<ast::astModule>(loc(), name);
    auto tok = next_token();
    auto inherit = std::make_unique<ast::astList>(loc());
    if (tok == token(':'))
    {
        do
        {
            auto id = parse_id();
            inherit->add(std::move(std::make_unique<ast::astExpr>(loc(), id)));
        } while (next_token() == token(','));
    }
    if (cur_tok() != token('{'))
    {
        THROW("Parse error: Expected '{' at " + loc().print());
    }
    auto ptr = parse_dict();
    ptr->add("_attr_", std::move(attrs));
    ptr->add("_inherit_", std::move(inherit));
    node->add_attr(std::move(ptr));
    return std::move(node);
}