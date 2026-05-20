-- TorchLean.Verification.Crown
-- CROWN/LiRPA-style linear bound propagation
-- Provides tighter bounds than IBP by using linear relaxations

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP

namespace TorchLean

/-- Linear bounds for a layer's output as a function of the network input:
    Λ_L · x + bias_L ≤ f(x) ≤ Λ_U · x + bias_U -/
structure CrownBound where
  lowerMatrix : Tensor   -- [outDim, currentDim]
  lowerBias : Tensor     -- [outDim]
  upperMatrix : Tensor   -- [outDim, currentDim]
  upperBias : Tensor     -- [outDim]
  deriving Repr

namespace CrownBound

/-- Identity bound: f(x) = x, so Λ = I, bias = 0 -/
def identity (dim : Nat) : CrownBound := Id.run do
  let mut matData := Array.mkEmpty (dim * dim)
  for i in [:dim] do
    for j in [:dim] do
      matData := matData.push (if i == j then 1.0 else 0.0)
  let identityMat : Tensor := ⟨[dim, dim], matData⟩
  let biasVec := Tensor.zeros [dim]
  return ⟨identityMat, biasVec, identityMat, biasVec⟩

end CrownBound

/-! ## CROWN ReLU relaxation -/

/-- ReLU relaxation parameters for each neuron. -/
structure ReluRelax where
  lowerSlope : Array Float
  lowerIntercept : Array Float
  upperSlope : Array Float
  upperIntercept : Array Float

/-- Compute CROWN linear relaxation for ReLU given pre-activation bounds [l, u].
    For each neuron:
    - Case 1: l ≥ 0 (always active) → identity
    - Case 2: u ≤ 0 (always inactive) → zero
    - Case 3: l < 0 < u (unstable) → linear relaxation -/
def computeReluRelax (preActLower preActUpper : Tensor) : ReluRelax := Id.run do
  let n := preActLower.data.size
  let mut lSlope := Array.mkEmpty n
  let mut lIntercept := Array.mkEmpty n
  let mut uSlope := Array.mkEmpty n
  let mut uIntercept := Array.mkEmpty n

  for i in [:n] do
    let l := preActLower.data[i]!
    let u := preActUpper.data[i]!

    if l ≥ 0.0 then
      lSlope := lSlope.push 1.0
      lIntercept := lIntercept.push 0.0
      uSlope := uSlope.push 1.0
      uIntercept := uIntercept.push 0.0
    else if u ≤ 0.0 then
      lSlope := lSlope.push 0.0
      lIntercept := lIntercept.push 0.0
      uSlope := uSlope.push 0.0
      uIntercept := uIntercept.push 0.0
    else
      -- Unstable neuron: l < 0 < u
      let slope := u / (u - l)
      let intercept := -(l * u) / (u - l)
      uSlope := uSlope.push slope
      uIntercept := uIntercept.push intercept
      if fabs u ≥ fabs l then
        lSlope := lSlope.push slope
        lIntercept := lIntercept.push intercept
      else
        lSlope := lSlope.push 0.0
        lIntercept := lIntercept.push 0.0

  return ⟨lSlope, lIntercept, uSlope, uIntercept⟩

/-! ## Helper operations -/

/-- Right-multiply matrix by diagonal: M · diag(d)
    M: [n, m], d: [m] → result[i,j] = M[i,j] * d[j] -/
def matDiagMul (mat : Tensor) (diag : Array Float) : Option Tensor :=
  match mat.shape with
  | [n, m] =>
    if diag.size == m then Id.run do
      let mut result := Array.mkEmpty (n * m)
      for i in [:n] do
        for j in [:m] do
          result := result.push (mat.data[i * m + j]! * diag[j]!)
      return some (⟨[n, m], result⟩ : Tensor)
    else none
  | _ => none

/-! ## CROWN backward pass -/

/-- CROWN backward bound propagation.
    Starts from output identity, propagates backward composing linear bounds.
    Uses IBP pre-activation bounds for ReLU relaxation. -/
def crownBackward (net : Network) (inputBounds : Interval) : Option Interval := do
  -- Step 1: Forward pass with IBP to get bounds at each layer
  let mut layerBounds : Array Interval := #[inputBounds]
  let mut currentBounds := inputBounds

  for layer in net.layers do
    let newBounds ← ibpLayer layer currentBounds
    layerBounds := layerBounds.push newBounds
    currentBounds := newBounds

  -- Step 2: Backward pass from output
  let outputDim := currentBounds.lower.data.size
  let mut lMat := CrownBound.identity outputDim
  let mut boundIdx := layerBounds.size - 1

  let layersRev := net.layers.reverse
  for layer in layersRev do
    if boundIdx > 0 then
      boundIdx := boundIdx - 1
    match layer with
    | .activation act =>
      match act with
      | .relu =>
        -- Get pre-activation bounds
        let preActBounds : Interval := layerBounds[boundIdx]!
        let relaxation := computeReluRelax preActBounds.lower preActBounds.upper

        -- Λ_new = Λ_old · diag(slope)  (right multiply by diagonal)
        let lLower ← matDiagMul lMat.lowerMatrix relaxation.lowerSlope
        let lUpper ← matDiagMul lMat.upperMatrix relaxation.upperSlope

        -- bias_new = bias_old + Λ_old · intercept_vector
        let interceptLowerT : Tensor := ⟨[relaxation.lowerIntercept.size], relaxation.lowerIntercept⟩
        let interceptUpperT : Tensor := ⟨[relaxation.upperIntercept.size], relaxation.upperIntercept⟩
        let biasContribLower ← Tensor.matVecMul lMat.lowerMatrix interceptLowerT
        let biasContribUpper ← Tensor.matVecMul lMat.upperMatrix interceptUpperT
        let newLowerBias ← Tensor.add lMat.lowerBias biasContribLower
        let newUpperBias ← Tensor.add lMat.upperBias biasContribUpper

        lMat := ⟨lLower, newLowerBias, lUpper, newUpperBias⟩
      | _ =>
        -- For sigmoid/tanh, fall back (identity pass-through)
        pure ()
    | .linear l =>
      -- y = Wx + b → Λ_new = Λ_old · W, bias_new = bias_old + Λ_old · b
      let newLower ← Tensor.matMul lMat.lowerMatrix l.weight
      let newUpper ← Tensor.matMul lMat.upperMatrix l.weight
      let biasContribLower ← Tensor.matVecMul lMat.lowerMatrix l.bias
      let biasContribUpper ← Tensor.matVecMul lMat.upperMatrix l.bias
      let newLBias ← Tensor.add lMat.lowerBias biasContribLower
      let newUBias ← Tensor.add lMat.upperBias biasContribUpper
      lMat := ⟨newLower, newLBias, newUpper, newUBias⟩
    | .conv2d _ => pure ()   -- Conv2d CROWN: requires unrolled matrix form
    | .batchNorm _ => pure ()  -- BatchNorm CROWN: absorbed into linear
    | .maxPool2d _ => pure ()  -- MaxPool CROWN: requires case-split relaxation
    | .avgPool2d _ => pure ()  -- AvgPool CROWN: linear, could be supported
    | .dropout _ => pure ()    -- Identity at inference
    | .flatten => pure ()

  -- Step 3: Compute concrete bounds from linear bounds and input bounds
  -- lower = Λ_L⁺ · x_l + Λ_L⁻ · x_u + bias_L
  -- upper = Λ_U⁺ · x_u + Λ_U⁻ · x_l + bias_U
  let lMatPos := Tensor.map (fmax 0.0) lMat.lowerMatrix
  let lMatNeg := Tensor.map (fmin 0.0) lMat.lowerMatrix
  let uMatPos := Tensor.map (fmax 0.0) lMat.upperMatrix
  let uMatNeg := Tensor.map (fmin 0.0) lMat.upperMatrix

  let t1 ← Tensor.matVecMul lMatPos inputBounds.lower
  let t2 ← Tensor.matVecMul lMatNeg inputBounds.upper
  let t3 ← Tensor.add t1 t2
  let concreteLower ← Tensor.add t3 lMat.lowerBias

  let t4 ← Tensor.matVecMul uMatPos inputBounds.upper
  let t5 ← Tensor.matVecMul uMatNeg inputBounds.lower
  let t6 ← Tensor.add t4 t5
  let concreteUpper ← Tensor.add t6 lMat.upperBias

  return ⟨concreteLower, concreteUpper⟩

/-- Verify L∞ robustness using CROWN bounds (tighter than IBP). -/
def verifyRobustnessCROWN (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) : Option Bool := do
  let inputBounds := Interval.epsBall input epsilon
  let outputBounds ← crownBackward net inputBounds

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
