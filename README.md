## LGF Compiler

Logic graph frame (LGF) is an experimental project that providing a frameworks to formulate logic statements or conceptions into directed graphs, and transforming the graphs based on customized rules into others. This project is used to explore the potiential of the graph equivalence in practice. 

### Design principle:

Objects or values are treated as edges of graphs and the operation to these edges are nodes. One nodes consumes several edges and may produce a new value to be consumed by other nodes. This architecture forms a directed graphs can represent a widely processes or conceptions. 

#### Values and Nodes
Based on this idea, the operation can accept values as inputs and create one value as an output (Multiple outputs cases are equivalent to one node connect to multiple nodes in parallel). Value is an abstract object standing for anything could exchange between nodes.  Values and nodes consist the abstract graph.

#### Descriptor
The role of value is determined by the description class `valueDesc`. It is used to store all the properties of the value it associated with. It also provides all the interface method needed for interacting with the values. Note that value descriptor can be customized and can be converted while manipulate the graph if needed.

#### Passes
After a graph is constructed, one can apply a sequence of processes on the graph to change or transform the graph to another. It includes inserting, replacing and erasing nodes and edges (connection beteween nodes). 

### `C++` interface

### Todo's

* Secure the `unitDesc` and `zeroDesc` is flatten, ie.  `unit<unit<real>> = unit<real>`.
* Improve the `unit/zero_effective_check`.
* Add fix region for `partialDifferentOp`.
* Add `option` object for pass.
* Fix the `convert2SIO` pass.
* Enable the `export2latex` in interface.
* Add transform to API graph to enable numerical calculation.