#ifndef LGF_AST_TREE_H
#define LGF_AST_TREE_H
#include <vector>
#include <memory>
#include <string>

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
        astNode(astType k) : kind(k) {}
        virtual ~astNode() = default;
        void set_kind(astType k) { kind = k; }

    private:
        astType kind;
    };

    class astBlock : public astNode
    {
    public:
        astBlock() : astNode(astType::block) {};
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

    class astExpr : public astNode
    {
    public:
        astExpr() : astNode(astType::expr) {};
        std::string get_expr() { return id; }

    private:
        std::string id = "";
    };

    class astNumber : public astNode
    {
    public:
        astNumber() : astNode(astType::number) {};
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
        astBinaryOp() : astNode(astType::call) {};
        astBinaryOpType get_op() { return op_type; }
        astNode *get_lhs() { return lhs; }
        astNode *get_rhs() { return rhs; }

    private:
        astBinaryOpType op_type;
        astNode *lhs, *rhs;
    };

    class astDefine : public astNode
    {
    public:
        astDefine() : astNode(astType::define) {};
        std::string get_name() { return name; }
        astBlock *get_value() { return block; }

    private:
        std::string name;
        astBlock *block;
    };

    class astModule : public astBlock
    {
    public:
        astModule(const std::string &n) : astBlock(), name(n)
        {
            set_kind(astType::module);
        };
        std::string get_name() { return name; }

    private:
        std::string name;
    };
}

#endif