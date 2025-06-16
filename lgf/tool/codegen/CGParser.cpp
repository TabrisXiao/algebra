
#include "CGParser.h"
using namespace ast;
using kind_t = ast::token::kind;
namespace codegen
{
    void CGParser::parse_dict_data(dictData *ptr)
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
                auto subdict = std::make_unique<astDictionary>(loc());
                parse_dict_data(dynamic_cast<dictData *>(subdict.get()));
                ptr->add(key, std::move(subdict));
                break;
            }
            case kind_t('['):
                ptr->add(key, std::move(parse_list()));
                break;
            case kind_t('<'):
                ptr->add(key, std::move(parse_set()));
                break;
            case kind_t::tok_identifier:
                ptr->add(key, std::move(std::make_unique<astExpr>(loc(), parse_id())));
                break;
            default:
                emit_error("Unknown dictionary item!");
                break;
            }
        }
        return;
    }
    std::unique_ptr<astContext> CGParser::parse_context()
    {
        // assuming key word 'context' is already parsed and the current char is '<' or a char.
        auto ctx = std::make_unique<astContext>(loc());
        if (cur_tok().is(kind_t('<')))
        {
            auto attrs = parse_set();
            ctx->add("_attr_", std::move(attrs));
        }
        auto id = parse_id();
        ctx->set_name(id);
        parse_left_brace();
        auto content = std::make_unique<astList>(loc());
        while (!cur_tok().is_any(kind_t('}'), kind_t::tok_eof))
        {
            auto key = parse_id();
            if (key == "module")
            {
                auto module = parse_module();
                auto name = module->get_name();
                content->add(std::move(module));
            }
            else if (key == "context")
            {
                auto sub = parse_context();
                auto name = sub->get<astExpr>("_name_")->string();
                content->add(std::move(sub));
            }
            else
            {
                emit_error("Unknown key word: " + key);
            }
        }
        ctx->add("_content_", std::move(content));
        return std::move(ctx);
    }

    std::unique_ptr<astContext> CGParser::parse()
    {
        // return true if error
        consume();
        auto id = parse_id();
        emit_error_if(id != "context", "Context must be the first keyword!");
        auto root = std::make_unique<astContext>(loc());
        return std::move(parse_context());
    }

    std::unique_ptr<astDictionary> CGParser::parse_dict()
    {
        // a dictionary doesn't allow duplicate keys
        // the syntax is {key1: value1, key2: value2, key3: value3...}
        auto node = std::make_unique<astDictionary>(loc());
        parse_dict_data(dynamic_cast<dictData *>(node.get()));
        return std::move(node);
    }

    std::unique_ptr<astList> CGParser::parse_list()
    {
        // a list allows duplicate items
        // the syntax is [item1, item2, item3...]
        parse_left_bracket();
        auto node = std::make_unique<astList>(loc());
        while (!cur_tok().is(kind_t(',')))
        {
            auto tok = cur_tok();
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

    std::unique_ptr<astList> CGParser::parse_set()
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

    std::unique_ptr<astModule> CGParser::parse_module()
    {
        // a module is a dictionary
        // the syntax is module<attr1, attr2, attr3...> name : inherit1, inherit2, inherit3...
        // { key1: value1, key2: value2, key3: value3...}

        auto attrs = parse_set();
        auto name = parse_id();
        auto node = std::make_unique<astModule>(loc(), name.c_str());
        auto tok = cur_tok();
        auto inherit = std::make_unique<astList>(loc());
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
        parse_dict_data(dynamic_cast<dictData *>(node.get()));
        return std::move(node);
    }
}
