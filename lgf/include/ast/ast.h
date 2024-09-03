
#ifndef AST_AST_H
#define AST_AST_H
#include <map>
#include <string>
#include <memory>
#include <vector>
#include "ast/lexer.h"
namespace ast
{
    using location = ::ast::charLocation;
    class astNode
    {
    public:
        enum kind
        {
            dict,
            list,
            expr,
            module,
            context,
            import,
        };
        astNode(location &l, const kind k) : k_(k), loc(l) {}
        virtual ~astNode() = default;
        kind get_kind() const { return k_; }
        location get_loc() const { return loc; }
        template <typename T>
        T *as() { return dynamic_cast<T *>(this); }

    private:
        const kind k_;
        location loc;
    };

    class listData
    {
    public:
        listData() = default;
        virtual ~listData() = default;
        void add(std::unique_ptr<astNode> ptr)
        {
            data.push_back(std::move(ptr));
        }
        template <typename T>
        T *get(size_t i = 0)
        {
            if (i < data.size())
                dynamic_cast<T *>(data[i].get());
            else
                return nullptr;
        }
        std::vector<std::unique_ptr<astNode>> &get_content() { return data; }
        size_t size() const { return data.size(); }

    private:
        std::vector<std::unique_ptr<astNode>> data;
    };

    class astList : public astNode, public listData
    {
    public:
        astList(location &lc) : astNode(lc, kind::list) {}
    };

    class dictData
    {
    public:
        dictData() = default;
        virtual ~dictData() = default;
        void add(const std::string &key, std::unique_ptr<astNode> ptr)
        {
            data[key] = std::move(ptr);
        }
        template <typename T>
        T *get(std::string key)
        {
            return dynamic_cast<T *>(data[key].get());
        }
        template <typename T>
        T *get(const char *key)
        {
            return dynamic_cast<T *>(data[key].get());
        }
        size_t size() const { return data.size(); }
        std::map<std::string, std::unique_ptr<astNode>> &get_content() { return data; }
        bool has(const std::string &key) const { return data.find(key) != data.end(); }

    private:
        std::map<std::string, std::unique_ptr<astNode>> data;
    };

    class astDictionary : public astNode, public dictData
    {
    public:
        astDictionary(location l) : astNode(l, kind::dict) {}
    };

    class astExpr : public astNode
    {
    public:
        astExpr(location l, const std::string &expr) : astNode(l, kind::expr), expr_(expr) {}
        std::string string() const { return expr_; }

    private:
        std::string expr_;
    };

    class astModule : public astNode, public dictData
    {
    public:
        astModule(location l, const char *_) : astNode(l, kind::module), name(_) {}
        std::string get_name()
        {
            return name;
        }
        astDictionary *attr(const char *key) { return dictData::get<astDictionary>(key); }

    private:
        const std::string name;
    };

    class astContext : public astNode, public dictData
    {
    public:
        astContext(location l) : astNode(l, kind::context) {}
        astContext(location l, const char *_) : astNode(l, kind::context), name(_) {}
        std::string get_name()
        {
            return name;
        }
        void set_name(std::string &n)
        {
            name = n;
        }
        void set_name(const char *n)
        {
            name = n;
        }

    private:
        std::string name = "";
    };
};

#endif