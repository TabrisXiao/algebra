
#include "libs/SIO/exporter.h"
#include "libs/Builtin/ops.h"
#include "libs/SIO/ops.h"

namespace lgf::SIO
{
    std::string export2latex::process(node *n)
    {
        std::string res;
        if (auto op = dynamic_cast<SIO::scalarProductOp *>(n))
        {
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_dual_node()) + "\\cdot ";
            }
            res = res.substr(0, res.size() - 6);
        }
        else if (auto op = dynamic_cast<sumOp *>(n))
        {
            res += "(";
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_dual_node()) + "+";
            }
            res.pop_back();
            res += ")";
        }
        else if (auto op = dynamic_cast<funcOp *>(n))
        {
            res += op->getFuncName() + "(";
            for (auto &input : op->get_input_handles())
            {
                res += process(input.get_dual_node()) + ",";
            }
            res.pop_back();
            res += ")";
        }
        else if (auto op = dynamic_cast<symbolOp *>(n))
        {
            auto sym = op->get_symbol();
            if (sym[0] == '%')
            {
                sym[0] = '{';
                sym = "x_" + sym + "}";
            }
            res += sym;
        }
        else if (auto op = dynamic_cast<partialD *>(n))
        {
            res = "\\frac{\\partial }{\\partial " + process(op->input(1)) + "} " + process(op->input(0));
        }
        else if (auto op = dynamic_cast<negativeOp *>(n))
        {
            res = "-" + process(op->input(0));
        }
        else if (auto op = dynamic_cast<differentialOp *>(n))
        {
            res = "d" + process(op->input(0)) + " ";
        }
        else
        {
            auto rep = "\n" + n->represent();
            THROW("The following Op is not supported in export2latex: " + rep + "\n");
        }
        return res;
    }

    // void export2Txt::run(graph* entry){
    //     auto list = entry->getNodeList();
    //     bool mainFound = false;
    //     auto module = dynamic_cast<moduleOp*>(entry);
    //     if(module && module->name == "main"){
    //         mainFound = true;
    //     }
    //     if( mainFound ) os<<"--- export main to text: ---\n\n";
    //     for(auto & node : list ){
    //         if(auto op = dynamic_cast<returnOp*>(node)){
    //             if(op->getInputSize() == 1){
    //                 auto val = op->inputValue(0);
    //                 os<<process(val);
    //             }
    //         }else if(auto module = dynamic_cast<moduleOp*>(node)){
    //             for(auto & input : module->getInputs()){
    //                 run(module);

    //             }
    //         }
    //     }
    //     if( mainFound ) os<<"\n\n--- end ---\n";
    // }

    // void export2latex::run(graph* entry){
    //     auto list = entry->getNodeList();
    //     bool mainFound = false;
    //     auto module = dynamic_cast<moduleOp*>(entry);
    //     if(module && module->name == "main"){
    //         mainFound = true;
    //     }
    //     if( mainFound ) os<<"--- export main to latex: ---\n\n";
    //     for(auto & node : list ){
    //         if(auto op = dynamic_cast<returnOp*>(node)){
    //             if(op->getInputSize() == 1){
    //                 auto val = op->inputValue(0);
    //                 os<<process(val);
    //             }
    //         }else if(auto module = dynamic_cast<moduleOp*>(node)){
    //             for(auto & input : module->getInputs()){
    //                 run(module);

    //             }
    //         }
    //     }
    //     if( mainFound ) os<<"\n\n--- end ---\n";
    // }

} // namespace lgf::SIO