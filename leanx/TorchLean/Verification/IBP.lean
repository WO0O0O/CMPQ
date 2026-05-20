-- TorchLean.Verification.IBP
-- Interval Bound Propagation for neural network verification

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers

namespace TorchLean

/-- Interval bounds for a tensor: each element has [lower, upper]. -/
structure Interval where
  lower : Tensor
  upper : Tensor
  deriving Repr, Inhabited

namespace Interval

/-- Create an interval from a single point (zero-width). -/
def point (t : Tensor) : Interval := ⟨t, t⟩

/-- Create an ε-ball around a point: [x - ε, x + ε]. -/
def epsBall (center : Tensor) (epsilon : Float) : Interval :=
  ⟨Tensor.map (· - epsilon) center, Tensor.map (· + epsilon) center⟩

/-- Check if the interval is valid (lower ≤ upper element-wise). -/
def isValid (iv : Interval) : Bool := Id.run do
  if iv.lower.data.size != iv.upper.data.size then return false
  for i in [:iv.lower.data.size] do
    if iv.lower.data[i]! > iv.upper.data[i]! then return false
  return true

/-- Width of the interval (upper - lower) at each element. -/
def width (iv : Interval) : Option Tensor := Tensor.sub iv.upper iv.lower

/-- Midpoint of the interval. -/
def midpoint (iv : Interval) : Option Tensor := do
  let sum ← Tensor.add iv.lower iv.upper
  return Tensor.scalarMul 0.5 sum

end Interval

/-! ## IBP propagation through layers -/

/-- Propagate interval bounds through a linear layer: y = Wx + b
    Using positive/negative weight decomposition:
      W⁺ = max(W, 0), W⁻ = min(W, 0)
      new_lower = W⁺ · lower + W⁻ · upper + b
      new_upper = W⁺ · upper + W⁻ · lower + b -/
def ibpLinear (layer : LinearLayer) (bounds : Interval) : Option Interval := do
  -- Decompose weight matrix into positive and negative parts
  let wPos := Tensor.map (fmax 0.0) layer.weight  -- W⁺
  let wNeg := Tensor.map (fmin 0.0) layer.weight  -- W⁻

  -- Compute lower bound: W⁺ · lower + W⁻ · upper + b
  let posLower ← Tensor.matVecMul wPos bounds.lower
  let negUpper ← Tensor.matVecMul wNeg bounds.upper
  let sumLower ← Tensor.add posLower negUpper
  let newLower ← Tensor.add sumLower layer.bias

  -- Compute upper bound: W⁺ · upper + W⁻ · lower + b
  let posUpper ← Tensor.matVecMul wPos bounds.upper
  let negLower ← Tensor.matVecMul wNeg bounds.lower
  let sumUpper ← Tensor.add posUpper negLower
  let newUpper ← Tensor.add sumUpper layer.bias

  return ⟨newLower, newUpper⟩

/-- Propagate interval bounds through ReLU activation.
    ReLU is monotone, so: [max(0, lower), max(0, upper)] -/
def ibpReLU (bounds : Interval) : Interval :=
  ⟨Tensor.reluT bounds.lower, Tensor.reluT bounds.upper⟩

/-- Propagate interval bounds through sigmoid activation.
    Sigmoid is monotone, so: [σ(lower), σ(upper)] -/
def ibpSigmoid (bounds : Interval) : Interval :=
  ⟨Tensor.map sigmoid bounds.lower, Tensor.map sigmoid bounds.upper⟩

/-- Propagate interval bounds through tanh activation.
    Tanh is monotone, so: [tanh(lower), tanh(upper)] -/
def ibpTanh (bounds : Interval) : Interval :=
  ⟨Tensor.map tanh' bounds.lower, Tensor.map tanh' bounds.upper⟩

/-- Propagate interval bounds through Conv2d using W⁺/W⁻ decomposition.
    Similar to linear IBP but operates on the convolution kernel. -/
def ibpConv2d (layer : Conv2dLayer) (bounds : Interval) : Option Interval :=
  match bounds.lower.shape with
  | [inC, inH, inW] =>
    if inC != layer.inChannels then none
    else Id.run do
      let outH := layer.outputH inH
      let outW := layer.outputW inW
      let outSize := layer.outChannels * outH * outW
      let mut lowerData := Array.mkEmpty outSize
      let mut upperData := Array.mkEmpty outSize

      for oc in [:layer.outChannels] do
        for oh in [:outH] do
          for ow in [:outW] do
            let mut sumLower := layer.bias.data[oc]!
            let mut sumUpper := layer.bias.data[oc]!
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
                    let w := layer.kernel.data[kernelIdx]!
                    let l := bounds.lower.data[inputIdx]!
                    let u := bounds.upper.data[inputIdx]!
                    -- W⁺/W⁻ decomposition
                    if w ≥ 0.0 then
                      sumLower := sumLower + w * l
                      sumUpper := sumUpper + w * u
                    else
                      sumLower := sumLower + w * u
                      sumUpper := sumUpper + w * l
            lowerData := lowerData.push sumLower
            upperData := upperData.push sumUpper

      let outShape := [layer.outChannels, outH, outW]
      return some ⟨⟨outShape, lowerData⟩, ⟨outShape, upperData⟩⟩
  | _ => none

/-- Propagate interval bounds through BatchNorm (inference mode).
    y = γ · (x − μ) / √(σ² + ε) + β
    This is an affine transformation, so use W⁺/W⁻ decomposition. -/
def ibpBatchNorm (layer : BatchNormLayer) (bounds : Interval) : Option Interval := do
  if bounds.lower.data.size != layer.numFeatures then none
  let mut lowerData := Array.mkEmpty layer.numFeatures
  let mut upperData := Array.mkEmpty layer.numFeatures

  for i in [:layer.numFeatures] do
    let g := layer.gamma.data[i]!
    let b := layer.beta.data[i]!
    let mean := layer.runningMean.data[i]!
    let var := layer.runningVar.data[i]!
    let scale := g / Float.sqrt (var + layer.epsilon)
    let offset := b - scale * mean
    let l := bounds.lower.data[i]!
    let u := bounds.upper.data[i]!
    -- scale · x + offset: affine with sign of scale
    if scale ≥ 0.0 then
      lowerData := lowerData.push (scale * l + offset)
      upperData := upperData.push (scale * u + offset)
    else
      lowerData := lowerData.push (scale * u + offset)
      upperData := upperData.push (scale * l + offset)

  return ⟨⟨bounds.lower.shape, lowerData⟩, ⟨bounds.upper.shape, upperData⟩⟩

/-- Propagate interval bounds through MaxPool2d.
    Max is monotone, so: lower[i] = max over window of lower, upper[i] = max over window of upper -/
def ibpMaxPool2d (layer : MaxPool2dLayer) (bounds : Interval) : Option Interval :=
  match bounds.lower.shape with
  | [c, inH, inW] => Id.run do
    let outH := layer.outputH inH
    let outW := layer.outputW inW
    let outSize := c * outH * outW
    let mut lowerData := Array.mkEmpty outSize
    let mut upperData := Array.mkEmpty outSize

    for ch in [:c] do
      for oh in [:outH] do
        for ow in [:outW] do
          let mut maxLower := -1.0e38
          let mut maxUpper := -1.0e38
          for kh in [:layer.kernelH] do
            for kw in [:layer.kernelW] do
              let ih := oh * layer.strideH + kh - layer.padH
              let iw := ow * layer.strideW + kw - layer.padW
              if ih < inH && iw < inW then
                let idx := ch * inH * inW + ih * inW + iw
                maxLower := fmax maxLower bounds.lower.data[idx]!
                maxUpper := fmax maxUpper bounds.upper.data[idx]!
          lowerData := lowerData.push maxLower
          upperData := upperData.push maxUpper

    let outShape := [c, outH, outW]
    return some ⟨⟨outShape, lowerData⟩, ⟨outShape, upperData⟩⟩
  | _ => none

/-- Propagate interval bounds through AvgPool2d.
    Average is a linear operation: lower = avg(lower), upper = avg(upper) -/
def ibpAvgPool2d (layer : AvgPool2dLayer) (bounds : Interval) : Option Interval :=
  match bounds.lower.shape with
  | [c, inH, inW] => Id.run do
    let outH := layer.outputH inH
    let outW := layer.outputW inW
    let poolSize := Float.ofNat (layer.kernelH * layer.kernelW)
    let outSize := c * outH * outW
    let mut lowerData := Array.mkEmpty outSize
    let mut upperData := Array.mkEmpty outSize

    for ch in [:c] do
      for oh in [:outH] do
        for ow in [:outW] do
          let mut sumL := 0.0
          let mut sumU := 0.0
          for kh in [:layer.kernelH] do
            for kw in [:layer.kernelW] do
              let ih := oh * layer.strideH + kh - layer.padH
              let iw := ow * layer.strideW + kw - layer.padW
              if ih < inH && iw < inW then
                let idx := ch * inH * inW + ih * inW + iw
                sumL := sumL + bounds.lower.data[idx]!
                sumU := sumU + bounds.upper.data[idx]!
          lowerData := lowerData.push (sumL / poolSize)
          upperData := upperData.push (sumU / poolSize)

    let outShape := [c, outH, outW]
    return some ⟨⟨outShape, lowerData⟩, ⟨outShape, upperData⟩⟩
  | _ => none

/-- Propagate interval bounds through a single layer. -/
def ibpLayer (layer : Layer) (bounds : Interval) : Option Interval :=
  match layer with
  | .linear l => ibpLinear l bounds
  | .activation act =>
    match act with
    | .relu    => some (ibpReLU bounds)
    | .sigmoid => some (ibpSigmoid bounds)
    | .tanh    => some (ibpTanh bounds)
  | .conv2d l => ibpConv2d l bounds
  | .batchNorm l => ibpBatchNorm l bounds
  | .maxPool2d l => ibpMaxPool2d l bounds
  | .avgPool2d l => ibpAvgPool2d l bounds
  | .dropout _ => some bounds  -- Identity at inference
  | .flatten => some bounds  -- Flatten preserves element values

/-- Propagate interval bounds through an entire network.
    Returns output bounds for the given input bounds. -/
def ibpNetwork (net : Network) (inputBounds : Interval) : Option Interval :=
  net.layers.foldlM (fun acc layer => ibpLayer layer acc) inputBounds

/-- Verify L∞ robustness: does the network maintain the same classification
    for all inputs within ε of the given input?
    Returns `some true` if verified robust, `some false` if not verified,
    `none` if computation failed. -/
def verifyRobustnessIBP (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) : Option Bool := do
  -- Create ε-ball around input
  let inputBounds := Interval.epsBall input epsilon

  -- Propagate through network
  let outputBounds ← ibpNetwork net inputBounds

  -- Check robustness: true class lower bound > all other upper bounds
  let trueClassLower ← outputBounds.lower.getFlat trueClass

  let numClasses := outputBounds.lower.data.size
  let mut robust := true
  for i in [:numClasses] do
    if i != trueClass then
      let otherUpper ← outputBounds.upper.getFlat i
      if otherUpper ≥ trueClassLower then
        robust := false
  return robust

end TorchLean
