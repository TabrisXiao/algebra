

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
    typedef std::unique_ptr<edge> edgeHandle;
    class edgeBundle : public std::vector<edgeHandle>
    {
        public:
        edgeBundle() = default;
        void need_clean()
        {
            bNeedClean = 1;
        }
        bool is_valid_handle(edgeHandle &e);

        void clean();
        edgeHandle& operator[](size_t i){
            clean();
            return std::vector<edgeHandle>::operator[](i);
        }
        edgeHandle& at(size_t i){
            clean();
            return std::vector<edgeHandle>::at(i);
        }
        size_t size(){
            clean();
            return std::vector<edgeHandle>::size();
        }
        void push_back(edgeHandle &e);

        private:
        bool bNeedClean = 0;
    };
    class edge
    {
    public:
        edge(node *n, edgeBundle* b=nullptr) : _n(n), bundle(b) {}
        edge(const edge &e) = delete;
        edge operator=(const edge &e) = delete;

        ~edge()
        {
            decouple();
        }

        void update_bundle(edgeBundle* b){
            if(bundle) bundle->need_clean();
            bundle = b;
        }

        void couple(edge *e)
        {
            if( dual ) dual->break_edge();
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
            bundle->need_clean();
            if (!dual)
                return;
            dual->break_edge();
            dual = nullptr;
        }

        void break_edge()
        {
            dual = nullptr;
            bundle->need_clean();
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
        edgeBundle* get_bundle(){
            return bundle;
        }
        // void print_info(){
        //     // debuging function
        //     std::cout<<"edge: "<<this<<"  --------------"<<std::endl;
        //     std::cout<<"    node: "<<_n<<std::endl;
        //     std::cout<<"    dual: "<<dual<<std::endl;
        // }

        static bool is_valid_handle(edgeHandle &e){
            return e && e->is_coupled();
        }

        edge(edge &&e)
        {
            std::cout<<"move edge"<<std::endl;
            _n = e.get_node();
            dual = e.get_dual_edge();
            if (!dual)
                return;
            dual->update_dual_edge(this);
            bundle = e.get_bundle();
            bundle->need_clean();
        }

    private:
        node *_n = nullptr;
        edge *dual = nullptr;
        edgeBundle *bundle= nullptr;
    };

} // namespace lgf

#endif