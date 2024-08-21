
#ifndef LIBS_CODEGEN_PARSER_H
#define LIBS_CODEGEN_PARSER_H
#include <set>
#include "ast/ast.h"
#include "ast/parser.h"
#include "ast/context.h"
namespace lgf::codegen
{
    class moduleParser : public ast::parser
    {
    public:
        using token = ast::cLikeLexer::cToken;
        moduleParser()
        {
            load_lexer<ast::cLikeLexer>();
        };
        virtual ~moduleParser() = default;
        token next_token()
        {
            return token(lx()->get_next_token());
        }

        std::unique_ptr<ast::astModule> parse(::ast::context &c, ::utils::fiostream &fs)
        {
            set_input_stream(fs);
            return parse_module(c);
        }

        std::unique_ptr<ast::astDictionary> parse_dict()
        {
            // a dictionary doesn't allow duplicate keys
            // the syntax is {key1: value1, key2: value2, key3: value3...}
            auto node = std::make_unique<ast::astDictionary>(loc());
            while (next_token() != token('}'))
            {
                if (cur_tok() != token::tok_identifier)
                {
                    THROW("Parse error: Expected identifier at " + loc().print());
                }
                auto key = get_string();
                parse_colon();
                auto tok = next_token();
                if (tok == token('{'))
                {
                    node->add(key, std::move(parse_dict()));
                }
                else if (tok == token::tok_identifier)
                {
                    node->add(key, std::move(std::make_unique<ast::astExpr>(loc(), get_string())));
                }
                else if (tok == token('['))
                {
                    node->add(key, std::move(parse_list()));
                }
                else if (tok == token('<'))
                {
                    node->add(key, std::move(parse_set()));
                }
                else if (tok == token::tok_number)
                {
                    node->add(key, std::move(std::make_unique<ast::astNumber>(loc(), get_number())));
                }
                else
                {
                    THROW("Parse error: Unknown dictionatry item at " + loc().print());
                }
            }
            return std::move(node);
        }

        std::unique_ptr<ast::astList> parse_list()
        {
            // a list allows duplicate items
            // the syntax is [item1, item2, item3...]
            auto node = std::make_unique<ast::astList>(loc());
            while (next_token() != token(','))
            {
                if (cur_tok() == token(']'))
                    break;
                if (cur_tok() == token::tok_identifier)
                {
                    node->add(std::move(std::make_unique<ast::astExpr>(loc(), get_string())));
                }
                else if (cur_tok() == token::tok_number)
                {
                    node->add(std::move(std::make_unique<ast::astNumber>(loc(), get_number())));
                }
                else
                {
                    THROW("Parse error: Unknown list item at " + loc().print());
                }
            }
            return std::move(node);
        }

        std::unique_ptr<ast::astList> parse_set()
        {
            // set is a list that can't have duplicate items
            // the syntax is <item1, item2, item3...>
            auto node = std::make_unique<ast::astList>(loc());
            std::set<std::string> key;
            while (next_token() != token(','))
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

        std::unique_ptr<ast::astModule> parse_module(::ast::context &ctx)
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
                while (next_token() == token(','))
                {
                    auto id = parse_id();
                    inherit->add(std::move(std::make_unique<ast::astExpr>(loc(), id)));
                }
            }
            else if (tok != token('{'))
            {
                THROW("Parse error: Expected '{' at " + loc().print());
            }
            auto ptr = parse_dict();
            ptr->add("_attr_", std::move(attrs));
            ptr->add("_inherit_", std::move(inherit));
            node->add_attr(std::move(ptr));
            return std::move(node);
        }
    };
}
#endif