#ifndef LGF_AST_TREE_H
#define LGF_AST_TREE_H
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "lgf/utils.h"
#include "utils/stream.h"

namespace lgf::ast
{
    enum astType : uint16_t
    {
        block = 1,
        expr = 2,
        number = 3,
        call = 4,
        define = 5,
        module = 6,
        variable = 7,
        dict = 8,
        list = 9,
    };
    enum astBinaryOpType : uint16_t
    {
        add = 1,
        sub = 2,
        mul = 3,
        div = 4,
        mod = 5,
        and_ = 6,
        or_ = 7,
        xor_ = 8,
        lshift = 9,
        rshift = 10,
        eq = 11,
        ne = 12,
        lt = 13,
        gt = 14,
        le = 15,
        ge = 16,
    };
    class astNode
    {
    public:
        astNode() = default;
        astNode(const ::utils::textLocation &l, astType k) : loc(l), kind(k) {}
        virtual ~astNode() = default;
        void set_kind(astType k) { kind = k; }
        astType get_kind() { return kind; }

    private:
        astType kind;
        ::utils::textLocation loc;
    };

    class astBlock : public astNode
    {
    public:
        astBlock(const ::utils::textLocation &loc) : astNode(loc, astType::block) {};
        std::vector<std::unique_ptr<astNode>> &get_nodes()
        {
            return nodes;
        }
        void add_node(std::unique_ptr<astNode> node)
        {
            nodes.push_back(std::move(node));
        }

    private:
        std::vector<std::unique_ptr<astNode>> nodes;
    };

    class astVar : public astNode
    {
    public:
        astVar(const ::utils::textLocation &loc) : astNode(loc, astType::variable) {};
        astVar(const ::utils::textLocation &loc, const std::string &t, const std::string &n) : astNode(loc, astType::variable), name(n), type(t) {}
        void set_name(const std::string &n) { name = n; }
        void set_type(const std::string &t) { type = t; }
        std::string get_type_id() { return type; }
        std::string get_name() { return name; }

    private:
        std::string name;
        std::string type;
    };

    class astExpr : public astNode
    {
    public:
        astExpr(const ::utils::textLocation &loc) : astNode(loc, astType::expr) {};
        astExpr(const ::utils::textLocation &loc, const std::string &e) : astNode(loc, astType::expr), id(e) {}
        void set_expr(const std::string &e) { id = e; }
        std::string get_expr() { return id; }

    private:
        std::string id = "";
    };

    class astNumber : public astNode
    {
    public:
        astNumber(const ::utils::textLocation &loc) : astNode(loc, astType::number) {};
        template <typename T>
        astNumber(const ::utils::textLocation &loc, T val) : astNode(loc, astType::number)
        {
            store(val);
        }
        template <typename T>
        void store(T val)
        {
            value = static_cast<uint64_t>(val);
        }
        template <typename T>
        T get()
        {
            return static_cast<T>(value);
        }

    private:
        uint64_t value = 0;
    };

    class astBinaryOp : public astNode
    {
    public:
        astBinaryOp(const ::utils::textLocation &loc) : astNode(loc, astType::call) {};
        astBinaryOpType get_op() { return op_type; }
        astNode *get_lhs() { return lhs; }
        astNode *get_rhs() { return rhs; }

    private:
        astBinaryOpType op_type;
        astNode *lhs, *rhs;
    };

    class astFuncDefine : public astNode
    {
    public:
        astFuncDefine(const ::utils::textLocation &loc) : astNode(loc, astType::define) {}
        std::string get_name() { return name; }
        std::unique_ptr<astBlock> &get_value() { return block; }
        void add_arg(std::unique_ptr<astNode> &&arg)
        {
            args.push_back(std::move(arg));
        }
        void add_ret(std::unique_ptr<astNode> &&r)
        {
            ret = std::move(r);
        }

    private:
        std::string name;
        std::vector<std::unique_ptr<astNode>> args;
        std::unique_ptr<astNode> ret;
        std::unique_ptr<astBlock> block;
    };

    class astDictionary : public astNode
    {
    public:
        astDictionary(const ::utils::textLocation &loc) : astNode(loc, astType::dict) {};
        logicResult add(const std::string &key, std::unique_ptr<astNode> &&node)
        {
            if (contents.find(key) != contents.end())
            {
                return logicResult::fail();
            }
            contents[key] = std::move(node);
            return logicResult::success();
        }
        logicResult find(const std::string &key)
        {
            if (contents.find(key) == contents.end())
            {
                return logicResult::fail();
            }
            return logicResult::success();
        }

        astNode *get(const std::string &key)
        {
            if (contents.find(key) == contents.end())
            {
                return nullptr;
            }
            return contents[key].get();
        }
        template <typename T>
        T *get(const std::string &key)
        {
            if (contents.find(key) == contents.end())
            {
                return nullptr;
            }
            return dynamic_cast<T *>(contents[key].get());
        }
        void erase(const std::string &key)
        {
            contents.erase(key);
        }
        std::map<std::string, std::unique_ptr<astNode>> &get_contents()
        {
            return contents;
        }

    private:
        std::map<std::string, std::unique_ptr<astNode>> contents;
    };

    class astList : public astNode
    {
    public:
        astList(const ::utils::textLocation &loc) : astNode(loc, astType::list) {};
        virtual ~astList() = default;
        logicResult add(std::unique_ptr<astNode> &&node)
        {
            if (node == nullptr)
            {
                return logicResult::fail();
            }
            nodes.push_back(std::move(node));
            return logicResult::success();
        }
        size_t size() { return nodes.size(); }
        template <typename T>
        T *get(size_t idx)
        {
            if (idx < nodes.size())
            {
                return dynamic_cast<T *>(nodes[idx].get());
            }
            return nullptr;
        }
        std::vector<std::unique_ptr<astNode>> &get_content()
        {
            return nodes;
        }

    private:
        std::vector<std::unique_ptr<astNode>> nodes;
    };

    class astModule : public astNode
    {
    public:
        astModule(const ::utils::textLocation &loc, const std::string &n) : astNode(loc, astType::module), name(n)
        {
            set_kind(astType::module);
        };
        std::string get_name() { return name; }
        astDictionary *get_attr() { return attr.get(); }
        std::vector<std::unique_ptr<astBlock>> &get_blocks() { return blocks; }
        void add_block(std::unique_ptr<astBlock> &&block)
        {
            blocks.push_back(std::move(block));
        }
        void add_attr(std::unique_ptr<astDictionary> &&a)
        {
            attr = std::move(a);
        }

    private:
        std::string name;
        std::unique_ptr<astDictionary> attr;
        std::vector<std::unique_ptr<astBlock>> blocks;
    };
}

#endif