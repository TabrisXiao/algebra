
#include "CGParser.h"
using namespace ast;
using kind_t = ast::token::kind;
namespace codegen
{
    void CGParser::parse_dict_data(CGContext *ctx, dictData *ptr)
    {
        parse_left_brace();
        while (try_consume(kind_t('}')).is_fail())
        {
            auto key = parse_id();
            parse_colon();
            switch (cur_tok().get_kind())
            {
            case kind_t('{'):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                auto subdict = std::make_unique<astDictionary>(loc());
                parse_dict_data(ctx, dynamic_cast<dictData *>(subdict.get()));
                ptr->add(key, std::move(subdict));
                break;
            }
            case kind_t('['):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                ptr->add(key, std::move(parse_list(ctx)));
                break;
            }
            case kind_t('<'):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                ptr->add(key, std::move(parse_set(ctx)));
                break;
            }
            case kind_t::tok_identifier:
            {
                ctx->create_symbol(key);
                ptr->add(key, std::move(std::make_unique<astExpr>(loc(), parse_id())));
                break;
            }
            default:
                emit_error("Unknown dictionary item!");
                break;
            }
        }
        return;
    }

    void CGParser::parse_module_region(CGContext *ctx, dictData *ptr)
    {
        parse_left_brace();
        while (try_consume(kind_t('}')).is_fail())
        {
            auto key = parse_id();
            parse_colon();
            if (key == "define")
            {
                ptr->add("_extra_", std::move(parse_literal()));
                continue;
            }
            switch (cur_tok().get_kind())
            {
            case kind_t('{'):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                auto subdict = std::make_unique<astDictionary>(loc());
                parse_dict_data(ctx, dynamic_cast<dictData *>(subdict.get()));
                ptr->add(key, std::move(subdict));
                break;
            }
            case kind_t('['):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                ptr->add(key, std::move(parse_list(ctx)));
                break;
            }
            case kind_t('<'):
            {
                CGContext::CGCGuard guard(ctx, key, symbolInfo(symbolInfo::kind_t::region));
                ptr->add(key, std::move(parse_set(ctx)));
                break;
            }
            case kind_t::tok_identifier:
                ctx->create_symbol(key);
                ptr->add(key, std::move(std::make_unique<astExpr>(loc(), parse_id())));
                break;
            default:
                emit_error("Unknown dictionary item!");
                break;
            }
        }
        return;
    }
    std::unique_ptr<astContext> CGParser::parse_context(CGContext *ctx)
    {
        // assuming key word 'context' is already parsed and the current char is '<' or a char.
        auto context = std::make_unique<astContext>(loc());
        if (cur_tok().is(kind_t('<')))
        {
            auto attrs = parse_set(ctx);
            context->add("_attr_", std::move(attrs));
        }
        auto id = parse_id();
        context->set_name(id);
        parse_left_brace();
        CGContext::CGCGuard guard(ctx, id, symbolInfo(symbolInfo::kind_t::context));
        auto content = std::make_unique<astList>(loc());
        while (try_consume(kind_t('}')).is_fail())
        {
            auto key = parse_id();
            if (key == "module")
            {
                auto module = parse_module(ctx);
                auto name = module->get_name();
                content->add(std::move(module));
            }
            else if (key == "context")
            {
                auto sub = parse_context(ctx);
                auto name = sub->get<astExpr>("_name_")->string();
                content->add(std::move(sub));
            }
            else
            {
                emit_error("Unknown key word: " + key);
            }
        }
        context->add("_content_", std::move(content));
        return std::move(context);
    }

    std::unique_ptr<astDictionary> CGParser::parse_dict(CGContext *ctx)
    {
        // a dictionary doesn't allow duplicate keys
        // the syntax is {key1: value1, key2: value2, key3: value3...}
        auto node = std::make_unique<astDictionary>(loc());
        parse_dict_data(ctx, dynamic_cast<dictData *>(node.get()));
        return std::move(node);
    }

    std::unique_ptr<astList> CGParser::parse_list(CGContext *ctx)
    {
        // a list allows duplicate items
        // the syntax is [item1, item2, item3...]
        parse_left_bracket();
        auto node = std::make_unique<astList>(loc());
        while (!cur_tok().is(kind_t(',')))
        {
            auto &tok = cur_tok();
            switch (tok.get_kind())
            {
            case kind_t(']'):
                consume();
                return std::move(node);
            case kind_t::tok_identifier:
                node->add(std::move(std::make_unique<astExpr>(loc(), tok.get_string())));
                break;
            case kind_t::tok_integer:
                node->add(std::move(std::make_unique<astExpr>(loc(), tok.get_string())));
                break;
            default:
                emit_error("Unknown list item!");
                break;
            }
            consume();
        }
        return std::move(node);
    }

    std::unique_ptr<astList> CGParser::parse_set(CGContext *ctx)
    {
        // set is a list that can't have duplicate items
        // the syntax is <item1, item2, item3...>
        parse_less_than();
        auto node = std::make_unique<astList>(loc());
        std::set<std::string> key;
        while (!cur_tok().is(kind_t(',')))
        {
            auto tok = cur_tok();
            switch (tok.get_kind())
            {
            case kind_t('>'):
                consume();
                return std::move(node);
            case kind_t::tok_identifier:
                if (!check_if_duplicate(node, tok.get_string()))
                {
                    node->add(std::move(std::make_unique<astExpr>(loc(), tok.get_string())));
                }
                break;
            case kind_t::tok_integer:
            case kind_t::tok_float:
                if (!check_if_duplicate(node, tok.get_string()))
                {
                    node->add(std::move(std::make_unique<astExpr>(loc(), tok.get_string())));
                }
                break;
            default:
                emit_error("Unknown set item!");
                break;
            }
            consume();
        }
        return std::move(node);
    }

    std::unique_ptr<astModule> CGParser::parse_module(CGContext *ctx)
    {
        // a module is a dictionary
        // the syntax is module<attr1, attr2, attr3...> name : inherit1, inherit2, inherit3...
        // { key1: value1, key2: value2, key3: value3...}

        auto attrs = parse_set(ctx);
        auto module_type = attrs->get<astExpr>()->string();
        auto name = parse_id();
        auto node = std::make_unique<astModule>(loc(), name.c_str());
        auto tok = cur_tok();
        auto inherit = std::make_unique<astList>(loc());
        symbolInfo::kind_t module_t;

        if (module_type == "attr")
        {
            module_t = symbolInfo::kind_t::attr;
        }
        else if (module_type == "node")
        {
            module_t = symbolInfo::kind_t::node;
        }
        else if (module_type == "desc")
        {
            module_t = symbolInfo::kind_t::desc;
        }
        else
        {
            emit_error("Unknown module type: " + module_type);
        }
        CGContext::CGCGuard guard(ctx, name, symbolInfo(module_t));
        if (try_consume(kind_t(':')).is_success())
        {
            do
            {
                auto id = parse_id();
                inherit->add(std::move(std::make_unique<astExpr>(loc(), id.data())));
            } while (try_consume(kind_t(',')).is_success());
        }

        node->add("_attr_", std::move(attrs));
        node->add("_parent_", std::move(inherit));
        parse_module_region(ctx, dynamic_cast<dictData *>(node.get()));
        return std::move(node);
    }
}

std::unique_ptr<astExpr> codegen::CGParser::parse_literal()
{
    parse_less_than();
    if (!cur_tok().is(kind_t('{')))
    {
        emit_error("Token '{' missing!");
    }
    std::string str = parse_literal_until("}>");

    str = str.substr(0, str.size() - 2);
    char lastChar = str[str.size() - 1];
    while (lastChar == ' ' || lastChar == '\n' || lastChar == '\t')
    {
        str.pop_back();
        lastChar = str[str.size() - 1];
    }
    return std::make_unique<astExpr>(loc(), str);
}