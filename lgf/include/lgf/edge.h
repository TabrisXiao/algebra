

#ifndef LGF_EDGE_H_
#define LGF_EDGE_H_
#include <vector>
#include <stdexcept>
namespace lgf
{
    class node;
    class edge
    {
    public:
        edge(node *n) : _n(n) {}
        edge(const edge &e) = delete;
        edge operator=(const edge &e) = delete;

        ~edge()
        {
            decouple();
        }

        void couple(edge *e)
        {
            dual = e;
            e->update_dual_edge(this);
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
        void decouple()
        {
            if (!dual)
                return;
            dual->break_edge();
            dual = nullptr;
        }

        void break_edge()
        {
            dual = nullptr;
        }

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
        // void print_info(){
        //     // debuging function
        //     std::cout<<"edge: "<<this<<"  --------------"<<std::endl;
        //     std::cout<<"    node: "<<_n<<std::endl;
        //     std::cout<<"    dual: "<<dual<<std::endl;
        // }

        edge(edge &&e)
        {
            _n = e.get_node();
            dual = e.get_dual_edge();
            if (!dual)
                return;
            dual->update_dual_edge(this);
        }

    private:
        node *_n = nullptr;
        edge *dual = nullptr;
    };

} // namespace lgf

#endif