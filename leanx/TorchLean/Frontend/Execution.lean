-- TorchLean.Frontend.Execution
-- Eager and Compiled execution modes (§3.1 of the paper)

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Frontend.Graph

namespace TorchLean

/-- Execution mode for model evaluation. -/
inductive ExecutionMode where
  | eager     -- Immediate computation, no graph construction
  | compiled  -- Lower to op-tagged SSA/DAG graph IR, then evaluate
  deriving Repr, BEq

/-- Data type specification for tensors. -/
inductive DataType where
  | float32   -- IEEE-754 binary32
  | float64   -- IEEE-754 binary64
  | int32     -- 32-bit integer
  | int64     -- 64-bit integer
  deriving Repr, BEq

/-- A model that can be executed in either eager or compiled mode. -/
structure ExecutableModel where
  network : Network
  mode : ExecutionMode
  inputShape : List Nat
  graph : Option ComputeGraph  -- Populated in compiled mode
  deriving Repr

namespace ExecutableModel

/-- Create a model in eager mode. -/
def eagerModel (net : Network) (inputShape : List Nat) : ExecutableModel :=
  ⟨net, .eager, inputShape, none⟩

/-- Create a model in compiled mode (builds graph IR). -/
def compiledModel (net : Network) (inputShape : List Nat) : ExecutableModel :=
  let graph := ComputeGraph.ofNetwork net inputShape
  ⟨net, .compiled, inputShape, some graph⟩

/-- Forward pass using the appropriate execution mode. -/
def forward (model : ExecutableModel) (input : Tensor) : Option Tensor :=
  match model.mode with
  | .eager => model.network.forward input
  | .compiled =>
    match model.graph with
    | some g => g.eval [input]
    | none => model.network.forward input  -- fallback

/-- Switch execution mode. -/
def setMode (model : ExecutableModel) (mode : ExecutionMode) : ExecutableModel :=
  match mode with
  | .eager => { model with mode := .eager }
  | .compiled =>
    let graph := ComputeGraph.ofNetwork model.network model.inputShape
    { model with mode := .compiled, graph := some graph }

/-- Get the computation graph (compile if needed). -/
def getGraph (model : ExecutableModel) : ComputeGraph :=
  match model.graph with
  | some g => g
  | none => ComputeGraph.ofNetwork model.network model.inputShape

end ExecutableModel

/-! ## ONNX-like Import Support

  Convert from a simplified ONNX-like representation to TorchLean Network. -/

/-- An ONNX-like operator specification. -/
inductive ONNXOp where
  | gemm (transA transB : Bool) (alpha beta : Float)  -- General matrix multiply
  | relu
  | sigmoid
  | tanh
  | conv (kernelShape : List Nat) (strides pads : List Nat)
  | batchNorm (epsilon : Float)
  | maxPool (kernelShape : List Nat) (strides pads : List Nat)
  | averagePool (kernelShape : List Nat) (strides pads : List Nat)
  | flatten (axis : Nat)
  | dropout (ratio : Float)
  | add       -- Element-wise addition
  | matMul    -- Matrix multiplication
  deriving Repr

/-- An ONNX-like graph node. -/
structure ONNXNode where
  name : String
  op : ONNXOp
  inputs : List String
  outputs : List String
  deriving Repr

/-- Look up a tensor by name in a list of named tensors. -/
private def lookupWeight (name : String) (weights : List (String × Tensor)) : Option Tensor :=
  weights.find? (fun p => p.1 == name) |>.map Prod.snd

/-- Convert an ONNX-like model to a TorchLean Network.
    Currently supports: Gemm (linear), ReLU, Sigmoid, Tanh. -/
def importONNX (nodes : List ONNXNode) (weights : List (String × Tensor)) : Option Network := do
  let mut net := Network.empty

  for node in nodes do
    match node.op with
    | .gemm _ _ _ _ =>
      -- Gemm → Linear layer
      let wName := node.inputs[1]!
      let bName := node.inputs[2]!
      let some w := lookupWeight wName weights | none
      let some b := lookupWeight bName weights | none
      match w.shape with
      | [outF, inF] =>
        net := net.addLayer (.linear ⟨inF, outF, w, b⟩)
      | _ => none
    | .relu => net := net.addLayer (.activation .relu)
    | .sigmoid => net := net.addLayer (.activation .sigmoid)
    | .tanh => net := net.addLayer (.activation .tanh)
    | .conv kernelShape strides pads =>
      let wName := node.inputs[1]!
      let bName := node.inputs[2]!
      let some w := lookupWeight wName weights | none
      let some b := lookupWeight bName weights | none
      match kernelShape, strides, pads with
      | [kH, kW], [sH, sW], [pH, _, _, _] =>
        match w.shape with
        | [outC, inC, _, _] =>
          net := net.addLayer (.conv2d ⟨inC, outC, kH, kW, sH, sW, pH, pH, w, b⟩)
        | _ => none
      | _, _, _ => pure ()
    | .batchNorm eps =>
      let some gamma := lookupWeight node.inputs[1]! weights | none
      let some beta := lookupWeight node.inputs[2]! weights | none
      let some mean := lookupWeight node.inputs[3]! weights | none
      let some var := lookupWeight node.inputs[4]! weights | none
      match gamma.shape with
      | [n] => net := net.addLayer (.batchNorm ⟨n, mean, var, gamma, beta, eps⟩)
      | _ => pure ()
    | .maxPool kernelShape strides pads =>
      match kernelShape, strides, pads with
      | [kH, kW], [sH, sW], [pH, _, _, _] =>
        net := net.addLayer (.maxPool2d ⟨kH, kW, sH, sW, pH, pH⟩)
      | _, _, _ => pure ()
    | .averagePool kernelShape strides pads =>
      match kernelShape, strides, pads with
      | [kH, kW], [sH, sW], [pH, _, _, _] =>
        net := net.addLayer (.avgPool2d ⟨kH, kW, sH, sW, pH, pH⟩)
      | _, _, _ => pure ()
    | .flatten _ => net := net.addLayer .flatten
    | .dropout rate => net := net.addLayer (.dropout ⟨rate⟩)
    | _ => pure ()  -- Skip unsupported ops

  return net

end TorchLean
