
#ifndef DTREE_H
#define DTREE_H
#include <vector>
#include <iostream>
#include <queue>
#include <stack>

#include "dgraph/include/dgraph.hpp"
namespace dgl
{
    class node : public vertex
    {
    public:
        node() = default;
        virtual ~node()
        {
            auto edges = getOutEdges();
            for (auto e : edges)
            {
                delete e;
            }
        }
        // link to a node, the level of the linked node will be set as this->level+1;
        // the linkTo function will create a new edge linking to the target node.
        template <typename N1, typename N2, typename... Ns>
        void addChild(N1 &n1, N2 &n2, Ns &...args)
        {
            addChild<N1>(n1);
            return addChild<N2, Ns...>(n2, args...);
        }
        template <typename N1>
        void addChild(N1 &n)
        {
            auto ed = new edge();
            ed->connect(*this, n);
            n.setLevel(level + 1);
            nextChildIndex = 0;
        }

        // deep first walk through all the node
        // fstart stands for the callalbe should be run when walk into a node at the first time
        // and the fend stands for the callable should be ran at the last access the node in the walk.
        template <typename callable1, typename callable2>
        void DFWalk(callable1 &fstart, callable2 &fend)
        {
            std::stack<node *> st;
            st.push(this);
            while (st.size())
            {
                auto n1 = st.top();
                if (!(n1->isExplored()))
                {
                    fstart(dynamic_cast<node*>(n1));
                    n1->setExplored(true);
                }
                auto child_node = n1->nextChild();
                if (child_node != nullptr)
                {
                    st.push(child_node);
                    continue;
                }
                fend(dynamic_cast<node*>(n1));
                n1->setExplored(false);
                st.pop();
            }
            return;
        }

        // get the parent node
        node *getParent()
        {   
            auto ptr = this->getInEdges()[0]->getSender();
            return dynamic_cast<node *>(ptr);
        }

        // get the next child node
        node *nextChild()
        {
            auto buf = getOutEdges();
            if (nextChildIndex >= buf.size())
            {
                nextChildIndex = 0;
                return nullptr;
            }       
            return dynamic_cast<node *>(buf[nextChildIndex++]->getReceiver());
        }

        void setLevel(uint64_t lv) { level = lv; }
        bool hasChild()
        {
            if (getOutEdges().size() == 0)
                return false;
            return true;
        }

        uint64_t level = 0;

    private:
        // this pointer is used internal for
        uint64_t nextChildIndex = 0;
    };
}

#endif