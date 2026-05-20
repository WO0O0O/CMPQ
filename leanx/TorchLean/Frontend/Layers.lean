-- TorchLean.Frontend.Layers
-- Neural network layer definitions with forward pass (§3.2 of the paper)

import TorchLean.Frontend.Tensor

namespace TorchLean

/-- Activation function type. -/
inductive ActivationFn where
  | relu
  | sigmoid
  | tanh
  deriving Repr, BEq

/-- Apply an activation function element-wise to a tensor. -/
def applyActivation (act : ActivationFn) (t : Tensor) : Tensor :=
  match act with
  | .relu    => Tensor.map relu t
  | .sigmoid => Tensor.map TorchLean.sigmoid t
  | .tanh    => Tensor.map tanh' t

/-! ## Linear (Fully Connected) Layer -/

/-- A linear (fully connected) layer: y = Wx + b
    weight: [outFeatures, inFeatures]
    bias:   [outFeatures] -/
structure LinearLayer where
  inFeatures : Nat
  outFeatures : Nat
  weight : Tensor   -- shape [outFeatures, inFeatures]
  bias : Tensor     -- shape [outFeatures]
  deriving Repr

namespace LinearLayer

/-- Create a linear layer with given weights and bias. -/
def mk' (inF outF : Nat) (weightData biasData : Array Float) : Option LinearLayer :=
  match Tensor.ofData [outF, inF] weightData, Tensor.ofData [outF] biasData with
  | some w, some b => some ⟨inF, outF, w, b⟩
  | _, _ => none

/-- Create a linear layer initialized to zeros. -/
def zeros (inF outF : Nat) : LinearLayer :=
  ⟨inF, outF, Tensor.zeros [outF, inF], Tensor.zeros [outF]⟩

/-- Forward pass: y = Wx + b -/
def forward (layer : LinearLayer) (input : Tensor) : Option Tensor := do
  let wx ← Tensor.matVecMul layer.weight input
  Tensor.add wx layer.bias

end LinearLayer

/-! ## Convolutional Layer (Conv2d) -/

/-- 2D convolution layer.
    kernel: [outChannels, inChannels, kernelH, kernelW]
    bias:   [outChannels]
    Note: Convolution is defined but marked as limited support per §9.1. -/
structure Conv2dLayer where
  inChannels : Nat
  outChannels : Nat
  kernelH : Nat
  kernelW : Nat
  strideH : Nat := 1
  strideW : Nat := 1
  padH : Nat := 0
  padW : Nat := 0
  kernel : Tensor   -- shape [outChannels, inChannels, kernelH, kernelW]
  bias : Tensor     -- shape [outChannels]
  deriving Repr

namespace Conv2dLayer

/-- Compute output spatial dimensions after convolution. -/
def outputH (layer : Conv2dLayer) (inputH : Nat) : Nat :=
  (inputH + 2 * layer.padH - layer.kernelH) / layer.strideH + 1

def outputW (layer : Conv2dLayer) (inputW : Nat) : Nat :=
  (inputW + 2 * layer.padW - layer.kernelW) / layer.strideW + 1

/-- Forward pass for Conv2d.
    input:  [inChannels, H, W]  (flattened in row-major order)
    output: [outChannels, outH, outW] -/
def forward (layer : Conv2dLayer) (input : Tensor) : Option Tensor :=
  match input.shape with
  | [inC, inH, inW] =>
    if inC != layer.inChannels then none
    else Id.run do
      let outH := layer.outputH inH
      let outW := layer.outputW inW
      let mut result := Array.mkEmpty (layer.outChannels * outH * outW)

      for oc in [:layer.outChannels] do
        for oh in [:outH] do
          for ow in [:outW] do
            let mut sum := layer.bias.data[oc]!
            for ic in [:layer.inChannels] do
              for kh in [:layer.kernelH] do
                for kw in [:layer.kernelW] do
                  let ih := oh * layer.strideH + kh - layer.padH
                  let iw := ow * layer.strideW + kw - layer.padW
                  if ih < inH && iw < inW then
                    let inputIdx := ic * inH * inW + ih * inW + iw
                    let kernelIdx := oc * layer.inChannels * layer.kernelH * layer.kernelW +
                                     ic * layer.kernelH * layer.kernelW +
                                     kh * layer.kernelW + kw
                    sum := sum + input.data[inputIdx]! * layer.kernel.data[kernelIdx]!
            result := result.push sum

      return some ⟨[layer.outChannels, outH, outW], result⟩
  | _ => none

end Conv2dLayer

/-! ## Batch Normalization -/

/-- Batch normalization layer (inference mode).
    y = (x − mean) / √(var + ε) · γ + β -/
structure BatchNormLayer where
  numFeatures : Nat
  runningMean : Tensor    -- [numFeatures]
  runningVar : Tensor     -- [numFeatures]
  gamma : Tensor          -- [numFeatures] (scale)
  beta : Tensor           -- [numFeatures] (shift)
  epsilon : Float := 1e-5
  deriving Repr

namespace BatchNormLayer

/-- Forward pass (inference mode only). -/
def forward (layer : BatchNormLayer) (input : Tensor) : Option Tensor :=
  match input.shape with
  | [n] =>
    if n != layer.numFeatures then none
    else Id.run do
      let mut result := Array.mkEmpty n
      for i in [:n] do
        let x := input.data[i]!
        let mean := layer.runningMean.data[i]!
        let var := layer.runningVar.data[i]!
        let g := layer.gamma.data[i]!
        let b := layer.beta.data[i]!
        let normalized := (x - mean) / Float.sqrt (var + layer.epsilon)
        result := result.push (g * normalized + b)
      return some ⟨[n], result⟩
  | _ => none

end BatchNormLayer

/-! ## Pooling Layers -/

/-- Max pooling 2D layer.
    Applies max operation over sliding window. -/
structure MaxPool2dLayer where
  kernelH : Nat
  kernelW : Nat
  strideH : Nat := 1
  strideW : Nat := 1
  padH : Nat := 0
  padW : Nat := 0
  deriving Repr

namespace MaxPool2dLayer

/-- Compute output spatial dimensions after pooling. -/
def outputH (layer : MaxPool2dLayer) (inputH : Nat) : Nat :=
  (inputH + 2 * layer.padH - layer.kernelH) / layer.strideH + 1

def outputW (layer : MaxPool2dLayer) (inputW : Nat) : Nat :=
  (inputW + 2 * layer.padW - layer.kernelW) / layer.strideW + 1

/-- Forward pass for MaxPool2d.
    input: [channels, H, W], output: [channels, outH, outW] -/
def forward (layer : MaxPool2dLayer) (input : Tensor) : Option Tensor :=
  match input.shape with
  | [c, inH, inW] => Id.run do
    let outH := layer.outputH inH
    let outW := layer.outputW inW
    let mut result := Array.mkEmpty (c * outH * outW)

    for ch in [:c] do
      for oh in [:outH] do
        for ow in [:outW] do
          let mut maxVal := -1.0e38
          for kh in [:layer.kernelH] do
            for kw in [:layer.kernelW] do
              let ih := oh * layer.strideH + kh - layer.padH
              let iw := ow * layer.strideW + kw - layer.padW
              if ih < inH && iw < inW then
                let idx := ch * inH * inW + ih * inW + iw
                maxVal := fmax maxVal input.data[idx]!
          result := result.push maxVal

    return some ⟨[c, outH, outW], result⟩
  | _ => none

end MaxPool2dLayer

/-- Average pooling 2D layer.
    Applies average operation over sliding window. -/
structure AvgPool2dLayer where
  kernelH : Nat
  kernelW : Nat
  strideH : Nat := 1
  strideW : Nat := 1
  padH : Nat := 0
  padW : Nat := 0
  deriving Repr

namespace AvgPool2dLayer

def outputH (layer : AvgPool2dLayer) (inputH : Nat) : Nat :=
  (inputH + 2 * layer.padH - layer.kernelH) / layer.strideH + 1

def outputW (layer : AvgPool2dLayer) (inputW : Nat) : Nat :=
  (inputW + 2 * layer.padW - layer.kernelW) / layer.strideW + 1

/-- Forward pass for AvgPool2d.
    input: [channels, H, W], output: [channels, outH, outW] -/
def forward (layer : AvgPool2dLayer) (input : Tensor) : Option Tensor :=
  match input.shape with
  | [c, inH, inW] => Id.run do
    let outH := layer.outputH inH
    let outW := layer.outputW inW
    let poolSize := Float.ofNat (layer.kernelH * layer.kernelW)
    let mut result := Array.mkEmpty (c * outH * outW)

    for ch in [:c] do
      for oh in [:outH] do
        for ow in [:outW] do
          let mut sum := 0.0
          for kh in [:layer.kernelH] do
            for kw in [:layer.kernelW] do
              let ih := oh * layer.strideH + kh - layer.padH
              let iw := ow * layer.strideW + kw - layer.padW
              if ih < inH && iw < inW then
                let idx := ch * inH * inW + ih * inW + iw
                sum := sum + input.data[idx]!
          result := result.push (sum / poolSize)

    return some ⟨[c, outH, outW], result⟩
  | _ => none

end AvgPool2dLayer

/-! ## Dropout Layer (Inference mode = identity) -/

/-- Dropout layer. At inference time, acts as identity. -/
structure DropoutLayer where
  rate : Float := 0.5
  deriving Repr

/-! ## Layer Type (Extended) -/

/-- A single layer in a neural network (extended with all layer types). -/
inductive Layer where
  | linear (l : LinearLayer)
  | conv2d (l : Conv2dLayer)
  | batchNorm (l : BatchNormLayer)
  | maxPool2d (l : MaxPool2dLayer)
  | avgPool2d (l : AvgPool2dLayer)
  | dropout (l : DropoutLayer)
  | activation (act : ActivationFn)
  | flatten  -- Flatten spatial dimensions
  deriving Repr

namespace Layer

/-- Forward pass for a single layer. -/
def forward (layer : Layer) (input : Tensor) : Option Tensor :=
  match layer with
  | .linear l => l.forward input
  | .conv2d l => l.forward input
  | .batchNorm l => l.forward input
  | .maxPool2d l => l.forward input
  | .avgPool2d l => l.forward input
  | .dropout _ => some input  -- Identity at inference
  | .activation act => some (applyActivation act input)
  | .flatten => some ⟨[input.data.size], input.data⟩

end Layer

/-! ## Sequential Network -/

/-- A sequential feedforward neural network. -/
structure Network where
  layers : List Layer
  deriving Repr

namespace Network

/-- Create an empty network. -/
def empty : Network := ⟨[]⟩

/-- Add a layer to the network. -/
def addLayer (net : Network) (l : Layer) : Network :=
  ⟨net.layers ++ [l]⟩

/-- Add a linear layer followed by an activation. -/
def addLinearAct (net : Network) (l : LinearLayer) (act : ActivationFn) : Network :=
  ⟨net.layers ++ [.linear l, .activation act]⟩

/-- Forward pass through the entire network. -/
def forward (net : Network) (input : Tensor) : Option Tensor :=
  net.layers.foldlM (fun acc layer => layer.forward acc) input

/-- Get all linear layers in the network. -/
def linearLayers (net : Network) : List LinearLayer :=
  net.layers.filterMap fun
    | .linear l => some l
    | _ => none

/-- Number of layers. -/
def numLayers (net : Network) : Nat := net.layers.length

/-- Total number of parameters. -/
def numParams (net : Network) : Nat :=
  net.layers.foldl (fun acc layer =>
    match layer with
    | .linear l => acc + l.weight.data.size + l.bias.data.size
    | .conv2d l => acc + l.kernel.data.size + l.bias.data.size
    | .batchNorm l => acc + l.gamma.data.size + l.beta.data.size +
                      l.runningMean.data.size + l.runningVar.data.size
    | .maxPool2d _ | .avgPool2d _ | .dropout _ | .activation _ | .flatten => acc) 0

end Network
end TorchLean
