

#ifndef LGF_EDGE_H_
#define LGF_EDGE_H_
#include <vector>
#include <stdexcept>
#include <memory>
#include <iostream>
namespace lgf
{
    class node;
    class edge;
    class edgeBundle : public std::vector<edge>
    {
    public:
        edgeBundle() = default;
        void need_clean()
        {
            bNeedClean = 1;
        }

        void clean();
        edge &operator[](size_t i)
        {
            // clean();
            return std::vector<edge>::operator[](i);
        }
        edge &at(size_t i)
        {
            clean();
            return std::vector<edge>::at(i);
        }
        size_t size()
        {
            clean();
            return std::vector<edge>::size();
        }
        size_t size0() { return std::vector<edge>::size(); }
        void push_back(edge &&e);

    private:
        bool bNeedClean = 0;
    };

    class edge
    {
    public:
        edge(node *n, edgeBundle *b = nullptr) : _n(n), bundle(b) {}
        edge(const edge &e) = delete;
        edge operator=(const edge &e) = delete;

        ~edge()
        {
            decouple();
        }

        void update_bundle(edgeBundle *b)
        {
            if (bundle)
                bundle->need_clean();
            bundle = b;
        }

        void reset()
        {
            if (bundle)
            {
                bundle->need_clean();
            }

            dual = nullptr;
            _n = nullptr;
        }

        void couple(edge &e)
        {
            if (dual)
                dual->reset();
            dual = &e;
            e.update_dual_edge(this);
        }

        void update_dual_edge(edge *e)
        {
            dual = e;
        }

        edge *get_dual_edge() const
        {
            return dual;
        }

        bool is_coupled() const
        {
            return dual != nullptr;
        }

        // empty this edge and update the dual to be empty as well
        void decouple();

        void update_node(node *n)
        {
            _n = n;
        }

        node *get_node() const
        {
            return _n;
        }

        node *get_dual_node() const
        {
            if (!dual)
                return nullptr;
            return dual->get_node();
        }

        edgeBundle *get_bundle()
        {
            return bundle;
        }

        edge(edge &) = delete;

        edge(edge &&e)
        {
            _n = e.get_node();
            auto target = e.get_dual_edge();
            if (target)
                couple(*(target));
            bundle = e.get_bundle();
            e.reset();
        }

        edge &operator=(edge &&e)
        {
            _n = e.get_node();
            auto target = e.get_dual_edge();
            if (target)
                couple(*(target));
            e.reset();
            return *this;
        }

    private:
        node *_n = nullptr;
        edge *dual = nullptr;
        edgeBundle *bundle = nullptr;
    };

} // namespace lgf

#endif