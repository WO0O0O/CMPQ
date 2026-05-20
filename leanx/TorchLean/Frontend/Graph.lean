-- TorchLean.Frontend.Graph
-- Op-tagged SSA/DAG computation graph IR

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers

namespace TorchLean

/-- Operations in the computation graph. -/
inductive Op where
  | input       -- Input node
  | matMul      -- Matrix-vector multiply
  | add         -- Element-wise addition
  | relu        -- ReLU activation
  | sigmoid     -- Sigmoid activation
  | tanh        -- Tanh activation
  | constant    -- Constant tensor
  deriving Repr, BEq

/-- A node in the computation graph (SSA form). -/
structure GraphNode where
  id : Nat
  op : Op
  inputs : List Nat       -- IDs of input nodes
  outputShape : List Nat
  constData : Option Tensor  -- For constant nodes
  deriving Repr

/-- Computation graph as a DAG of nodes. -/
structure ComputeGraph where
  nodes : Array GraphNode
  inputIds : List Nat
  outputId : Nat
  deriving Repr

namespace ComputeGraph

/-- Create an empty computation graph. -/
def empty : ComputeGraph := ⟨#[], [], 0⟩

/-- Add a node to the graph and return (updated graph, node id). -/
def addNode (g : ComputeGraph) (op : Op) (inputs : List Nat)
    (shape : List Nat) (constData : Option Tensor := none) : ComputeGraph × Nat :=
  let id := g.nodes.size
  let node : GraphNode := ⟨id, op, inputs, shape, constData⟩
  (⟨g.nodes.push node, g.inputIds, id⟩, id)

/-- Compile a Network into a ComputeGraph. -/
def ofNetwork (net : Network) (inputShape : List Nat) : ComputeGraph := Id.run do
  let mut g := ComputeGraph.empty
  -- Add input node
  let (g', inputId) := g.addNode .input [] inputShape
  g := { g' with inputIds := [inputId] }
  let mut currentId := inputId
  let mut currentShape := inputShape

  for layer in net.layers do
    match layer with
    | .linear l =>
      -- Add weight constant node
      let (g1, wId) := g.addNode .constant [] l.weight.shape (some l.weight)
      g := g1
      -- Add bias constant node
      let (g2, bId) := g.addNode .constant [] l.bias.shape (some l.bias)
      g := g2
      -- Add matmul node
      let outShape := [l.outFeatures]
      let (g3, mmId) := g.addNode .matMul [wId, currentId] outShape
      g := g3
      -- Add bias-add node
      let (g4, addId) := g.addNode .add [mmId, bId] outShape
      g := g4
      currentId := addId
      currentShape := outShape
    | .activation act =>
      let op := match act with
        | .relu    => Op.relu
        | .sigmoid => Op.sigmoid
        | .tanh    => Op.tanh
      let (g', actId) := g.addNode op [currentId] currentShape
      g := g'
      currentId := actId
    | .conv2d _ => pure ()   -- Conv2d graph lowering: future work
    | .batchNorm _ => pure () -- BatchNorm graph lowering: future work
    | .maxPool2d _ => pure () -- MaxPool graph lowering: future work
    | .avgPool2d _ => pure () -- AvgPool graph lowering: future work
    | .dropout _ => pure ()   -- Dropout is identity at inference
    | .flatten =>
      let flatShape := [currentShape.foldl (· * ·) 1]
      let (g', fId) := g.addNode .input [currentId] flatShape  -- passthrough
      g := g'
      currentId := fId
      currentShape := flatShape

  return { g with outputId := currentId }

/-- Evaluate a computation graph given input tensors. -/
def eval (g : ComputeGraph) (inputs : List Tensor) : Option Tensor := Id.run do
  -- Map from node ID to computed tensor
  let mut values : Array (Option Tensor) := Array.replicate g.nodes.size none

  -- Assign inputs
  let mut inputIdx := 0
  for nodeId in g.inputIds do
    if inputIdx < inputs.length then
      if nodeId < values.size then
        values := values.set! nodeId (some inputs[inputIdx]!)
      inputIdx := inputIdx + 1

  -- Process nodes in topological order (IDs are already sorted)
  for node in g.nodes do
    match node.op with
    | .input => pure () -- Already assigned
    | .constant =>
      values := values.set! node.id node.constData
    | .matMul =>
      match node.inputs with
      | [matId, vecId] =>
        match values[matId]!, values[vecId]! with
        | some mat, some vec =>
          values := values.set! node.id (Tensor.matVecMul mat vec)
        | _, _ => pure ()
      | _ => pure ()
    | .add =>
      match node.inputs with
      | [aId, bId] =>
        match values[aId]!, values[bId]! with
        | some a, some b =>
          values := values.set! node.id (Tensor.add a b)
        | _, _ => pure ()
      | _ => pure ()
    | .relu =>
      match node.inputs with
      | [inId] =>
        match values[inId]! with
        | some t => values := values.set! node.id (some (Tensor.reluT t))
        | none => pure ()
      | _ => pure ()
    | .sigmoid =>
      match node.inputs with
      | [inId] =>
        match values[inId]! with
        | some t => values := values.set! node.id (some (Tensor.map TorchLean.sigmoid t))
        | none => pure ()
      | _ => pure ()
    | .tanh =>
      match node.inputs with
      | [inId] =>
        match values[inId]! with
        | some t => values := values.set! node.id (some (Tensor.map tanh' t))
        | none => pure ()
      | _ => pure ()

  -- Return output
  if g.outputId < values.size then
    values[g.outputId]!
  else
    none

/-- Get the number of nodes in the graph. -/
def numNodes (g : ComputeGraph) : Nat := g.nodes.size

/-- Pretty-print the graph structure. -/
def toString (g : ComputeGraph) : String := Id.run do
  let mut s := s!"ComputeGraph ({g.numNodes} nodes):\n"
  for node in g.nodes do
    let opStr := match node.op with
      | .input    => "Input"
      | .matMul   => "MatMul"
      | .add      => "Add"
      | .relu     => "ReLU"
      | .sigmoid  => "Sigmoid"
      | .tanh     => "Tanh"
      | .constant => "Const"
    s := s ++ s!"  [{node.id}] {opStr} inputs={node.inputs} shape={node.outputShape}\n"
  s := s ++ s!"  output: node {g.outputId}\n"
  return s

instance : ToString ComputeGraph := ⟨ComputeGraph.toString⟩

end ComputeGraph
end TorchLean
