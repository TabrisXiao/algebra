## LGF Compiler

Logic graph frame (LGF) is an experimental project that providing a frameworks to formulate logic statements or conceptions into directed graphs, and transforming the graphs based on customized rules into others. This project is used to explore the potiential of the graph equivalence in practice. 

### Design principle:

Objects or values are treated as edges of graphs and the operation to these edges are nodes. One nodes consumes several edges and may produce a new value to be consumed by other nodes. This architecture forms a directed graphs can represent a widely processes or conceptions. 

#### Values and Operations
Based on this idea, the operation can accept values as inputs and create one value as an output (although it can create multiple outputs, but this case equivalents to one operation followed by multiple operations which producing one output each ). 

#### Types
Beside the object `value` and `operation` as building block to form a graph, the `type` of the value is an object used to describe what the value is and how we interface with this value. As a value produced from an operation, the properties of the value should be determined sololy based on that operation, so the type shouldn't be changed after it has been created. So we allow the copy constructor for `type` but should prevent any changes on the type itself. If you need a different type, you need to create a new one.