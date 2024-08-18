#ifndef LGF_CODEGEN_MODULE_H
#define LGF_CODEGEN_MODULE_H
#include "ast/ast.h"
#include "ast/generator.h"
#include "ast/lexer.h"
#include <map>
#include <memory>
#include "utils/stream.h"
#include "uid.h"
#include "ast/context.h"

using namespace lgf::ast;
namespace lgf::codegen
{
    class parserModule : public ast::parser
    {
    public:
        parserModule() = default;
        virtual ~parserModule() = default;
        ast::cLikeLexer::cToken next_token()
        {
            return ast::cLikeLexer::cToken(lx()->get_next_token());
        }
        virtual std::unique_ptr<astNode> parse(::ast::context &, fiostream &fs) = 0;
    };

    class parserBook
    {
    public:
        using paser_ptr = std::unique_ptr<astNode> (*)(fiostream &);
        parserBook() = default;
        ~parserBook() = default;
        parserModule *get(const std::string &id)
        {
            if (pmap.find(id) == pmap.end())
            {
                return nullptr;
            }
            return pmap[id].get();
        }
        std::map<std::string, std::unique_ptr<parserModule>> pmap;
    };

    class nodeParser : public parserModule
    {
        using token = cLikeLexer::cToken;

    public:
        nodeParser()
        {
            load_lexer<cLikeLexer>();
        }
        virtual ~nodeParser() = default;
        virtual std::unique_ptr<astNode> parse(::ast::context &c_, fiostream &fs) override
        {
            ctx = &c_;
            set_input_stream(fs);
            auto id = parse_id();
            auto idnode = std::make_unique<astExpr>(id);
            root = std::make_unique<astDictionary>();
            root->add("name", std::move(idnode));
            auto uid = std::make_unique<astNumber>(uid::uid_node);
            root->add("uid", std::move(uid));
            lx()->get_next_token();
            if (cur_tok() == cLikeLexer::cToken(':'))
            {
                auto pid = parse_id();
                root->add("parent", std::move(std::make_unique<astExpr>(pid)));
                lx()->get_next_token();
            }
            else if (cur_tok() != cLikeLexer::cToken('{'))
            {
                THROW("Parse error: Expecting '{' or ':'");
            }
            while (get_cur_char() != '}')
            {
                auto key = parse_id();
                if (key == "input")
                {
                    parse_input();
                }
                else if (key == "output")
                {
                    parse_output();
                }
                else if (key == "property")
                {
                    parse_property();
                }
                else if (key == "ir_name")
                {
                    parse_dict_item(key, root.get());
                }
                else
                {
                    THROW("Parse error: Unknown key: " + key);
                }
            }
            parse_right_brace();
            return std::move(root);
        }

        void parse_dict_item(std::string key, ast::astDictionary *node)
        {
            if (node->find(key).is_success())
            {
                THROW("Parse error: Duplicate key: " + key);
            }
            parse_equal();
            auto tok = next_token();
            if (tok == token('{'))
            {
                auto ptr = std::make_unique<astDictionary>();
                auto id = parse_id();
                parse_dict_item(id, ptr.get());
                parse_right_brace();
                node->add(key, std::move(ptr));
            }
            else
            {
                if (tok == token::tok_identifier)
                {
                    auto id = get_string();
                    node->add(key, std::move(std::make_unique<astExpr>(id)));
                }
                else
                {
                    THROW("Parse error: Unkown item at " + loc().print());
                }
            }
        }

        void parse_input()
        {
            THROW_WHEN(root->find("input").is_success(), "Parse error: Duplicate input.");
            auto ptr = std::make_unique<astDictionary>();
            parse_equal();
            parse_left_brace();
            do
            {
                auto type = parse_id();
                if (get_cur_char() == '*')
                {
                    parse_star();
                    type = type + "*";
                }
                parse_colon();
                auto id = parse_id();
                if (ptr->add(id, std::move(std::make_unique<astVar>(type, id))).is_fail())
                {
                    THROW("Parse error: Duplicate input signature:" + id);
                }
                if (get_cur_char() == ',')
                    parse_comma();
            } while (cur_tok() == token(token(',')));
            parse_right_brace();
            root->add("input", std::move(ptr));
        }
        void parse_output()
        {
            THROW_WHEN(root->find("output").is_success(), "Parse error: Duplicate output.");
            parse_equal();
            auto type = parse_id();
            parse_colon();
            auto id = parse_id();
            root->add("output", std::move(std::make_unique<astVar>(type, id)));
        }
        void parse_property()
        {
            THROW_WHEN(root->find("property").is_success(), "Parse error: Duplicate property.");
            auto ptr = std::make_unique<astList>();
            parse_equal();
            parse_left_brace();
            do
            {
                auto id = parse_id();
                if (ptr->add(std::move(std::make_unique<astExpr>(id))).is_fail())
                {
                    THROW("Parse error: Duplicate property:" + id);
                }
            } while (lx()->get_next_token() == token(token(',')));
            if (cur_tok() != token(token('}')))
            {
                THROW("Parse error: Expecting '}' at " + loc().print());
            }
            root->add("property", std::move(ptr));
        }

    private:
        std::unique_ptr<ast::astDictionary> root;
        ::ast::context *ctx;
    };
}

#endif