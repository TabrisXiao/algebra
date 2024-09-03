#ifndef CODEGEN_DEPENDENCY_H
#define CODEGEN_DEPENDENCY_H
#include <string>
#include <map>
#include <vector>
namespace codegen
{
    class dependenecySet
    {
    public:
        dependenecySet() = default;
        dependenecySet(std::string n, std::initializer_list<std::string> list)
            : name(n), content(list) {}
        std::string name;
        std::vector<std::string> content;
    };

    class dependencyMap
    {
    public:
        dependencyMap()
        {
            depMap["node"] = dependenecySet("node", {"lgf/node.h"});
            depMap["graph"] = dependenecySet("node", {"lgf/node.h"});
            depMap["attr"] = dependenecySet("attr", {"lgf/attribute.h"});
            depMap["identifier"] = dependenecySet("pass", {"lgf/pass.h"});
            depMap["normalizer"] = dependenecySet("pass", {"lgf/pass.h"});
        }
        void include(const char *skey)
        {
            auto dset = depMap.find(skey);
            if (dset == depMap.end())
            {
                return;
            }
            auto key = dset->second.name;
            auto it = std::find(basker_helper.begin(), basker_helper.end(), key);
            if (it != basker_helper.end())
                return;
            basker_helper.push_back(key);
            for (auto &dep : dset->second.content)
            {
                basket.push_back(dep);
            }
        }
        std::vector<std::string> get()
        {
            return basket;
        }
        void flush()
        {
            basket.clear();
            basker_helper.clear();
        }
        std::map<std::string, dependenecySet> depMap;
        std::vector<std::string> basket;
        std::vector<std::string> basker_helper;
    };
}

#endif // CODEGEN_DEPENDENCY_H
