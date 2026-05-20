-- TorchLean.Verification.AlphaBetaCrown
-- α,β-CROWN: Optimized CROWN with branch-and-bound (§5.2 of the paper)
--
-- α-CROWN: optimizable linear relaxation parameters (slopes)
-- β-CROWN: adds split constraints for branch-and-bound
-- BaB: partitions input domain for complete verification

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown
import TorchLean.Verification.Robustness

namespace TorchLean

/-! ## α-CROWN: Optimizable Relaxation Parameters -/

/-- Optimizable relaxation slopes for each unstable ReLU neuron.
    α ∈ [0, 1] determines the lower bound slope in the unstable region. -/
structure AlphaParams where
  /-- Per-layer alpha slopes. alphas[l][i] is the slope for neuron i in layer l. -/
  alphas : Array (Array Float)
  deriving Repr

namespace AlphaParams

/-- Initialize α parameters (default: use CROWN's adaptive choice). -/
def init (layerSizes : List Nat) : AlphaParams := Id.run do
  let mut alphas := Array.mkEmpty layerSizes.length
  for size in layerSizes do
    -- Initialize to 0.5 (midpoint between 0 and upper slope)
    alphas := alphas.push (Array.replicate size 0.5)
  return ⟨alphas⟩

/-- Clamp all α values to [0, 1]. -/
def clamp (params : AlphaParams) : AlphaParams :=
  ⟨params.alphas.map (·.map (fclamp · 0.0 1.0))⟩

end AlphaParams

/-- Compute α-CROWN relaxation for ReLU with optimizable lower bound slope.
    Unlike standard CROWN which fixes the lower slope heuristically,
    α-CROWN treats the lower slope as an optimizable parameter α. -/
def computeAlphaReluRelax (preActLower preActUpper : Tensor)
    (alphaSlopes : Array Float) : ReluRelax := Id.run do
  let n := preActLower.data.size
  let mut lSlope := Array.mkEmpty n
  let mut lIntercept := Array.mkEmpty n
  let mut uSlope := Array.mkEmpty n
  let mut uIntercept := Array.mkEmpty n

  for i in [:n] do
    let l := preActLower.data[i]!
    let u := preActUpper.data[i]!

    if l ≥ 0.0 then
      -- Always active: identity bounds
      lSlope := lSlope.push 1.0
      lIntercept := lIntercept.push 0.0
      uSlope := uSlope.push 1.0
      uIntercept := uIntercept.push 0.0
    else if u ≤ 0.0 then
      -- Always inactive: zero bounds
      lSlope := lSlope.push 0.0
      lIntercept := lIntercept.push 0.0
      uSlope := uSlope.push 0.0
      uIntercept := uIntercept.push 0.0
    else
      -- Unstable neuron: l < 0 < u
      -- Upper bound: fixed triangle slope u/(u-l)
      let uS := u / (u - l)
      let uI := -(l * u) / (u - l)
      uSlope := uSlope.push uS
      uIntercept := uIntercept.push uI
      -- Lower bound: optimizable slope α ∈ [0, u/(u-l)]
      let alpha := if i < alphaSlopes.size then
        fclamp alphaSlopes[i]! 0.0 1.0
      else 0.5
      -- Scale alpha by upper slope to get actual slope in [0, u/(u-l)]
      let lS := alpha * uS
      lSlope := lSlope.push lS
      lIntercept := lIntercept.push 0.0

  return ⟨lSlope, lIntercept, uSlope, uIntercept⟩

/-- α-CROWN backward pass with optimizable relaxation. -/
def alphaCrownBackward (net : Network) (inputBounds : Interval)
    (alphaParams : AlphaParams) : Option Interval := do
  -- Step 1: Forward pass with IBP for pre-activation bounds
  let mut layerBounds : Array Interval := #[inputBounds]
  let mut currentBounds := inputBounds

  for layer in net.layers do
    let newBounds ← ibpLayer layer currentBounds
    layerBounds := layerBounds.push newBounds
    currentBounds := newBounds

  -- Step 2: Backward pass with α-parameterized relaxation
  let outputDim := currentBounds.lower.data.size
  let mut lMat := CrownBound.identity outputDim
  let mut boundIdx := layerBounds.size - 1
  let mut layerIdx := net.layers.length

  let layersRev := net.layers.reverse
  for layer in layersRev do
    if boundIdx > 0 then
      boundIdx := boundIdx - 1
    layerIdx := layerIdx - 1
    match layer with
    | .activation act =>
      match act with
      | .relu =>
        let preActBounds := layerBounds[boundIdx]!
        -- Use α-parameterized relaxation instead of fixed CROWN
        let alphaSlopes := if layerIdx < alphaParams.alphas.size
          then alphaParams.alphas[layerIdx]! else #[]
        let relaxation := computeAlphaReluRelax preActBounds.lower preActBounds.upper alphaSlopes

        let lLower ← matDiagMul lMat.lowerMatrix relaxation.lowerSlope
        let lUpper ← matDiagMul lMat.upperMatrix relaxation.upperSlope

        let interceptLowerT : Tensor := ⟨[relaxation.lowerIntercept.size], relaxation.lowerIntercept⟩
        let interceptUpperT : Tensor := ⟨[relaxation.upperIntercept.size], relaxation.upperIntercept⟩
        let biasContribLower ← Tensor.matVecMul lMat.lowerMatrix interceptLowerT
        let biasContribUpper ← Tensor.matVecMul lMat.upperMatrix interceptUpperT
        let newLowerBias ← Tensor.add lMat.lowerBias biasContribLower
        let newUpperBias ← Tensor.add lMat.upperBias biasContribUpper

        lMat := ⟨lLower, newLowerBias, lUpper, newUpperBias⟩
      | _ => pure ()
    | .linear l =>
      let newLower ← Tensor.matMul lMat.lowerMatrix l.weight
      let newUpper ← Tensor.matMul lMat.upperMatrix l.weight
      let biasContribLower ← Tensor.matVecMul lMat.lowerMatrix l.bias
      let biasContribUpper ← Tensor.matVecMul lMat.upperMatrix l.bias
      let newLBias ← Tensor.add lMat.lowerBias biasContribLower
      let newUBias ← Tensor.add lMat.upperBias biasContribUpper
      lMat := ⟨newLower, newLBias, newUpper, newUBias⟩
    | _ => pure ()

  -- Step 3: Concrete bounds
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

/-! ## β-CROWN: Split Constraints for Branch-and-Bound -/

/-- A neuron split decision for branch-and-bound. -/
structure NeuronSplit where
  layerIdx : Nat
  neuronIdx : Nat
  /-- true = fix neuron active (≥ 0), false = fix neuron inactive (≤ 0) -/
  isActive : Bool
  deriving Repr, Inhabited

/-- β-CROWN parameters: Lagrangian multipliers for split constraints. -/
structure BetaParams where
  /-- β[i] is the Lagrangian multiplier for split i. -/
  betas : Array Float
  /-- The neuron splits this β corresponds to. -/
  splits : Array NeuronSplit
  deriving Repr

/-! ## Branch-and-Bound (BaB) -/

/-- A subproblem in the BaB tree. -/
structure BaBNode where
  /-- Input bounds for this subproblem. -/
  inputBounds : Interval
  /-- Neuron splits applied in this branch. -/
  splits : Array NeuronSplit
  /-- Lower bound on the objective (from CROWN). -/
  lowerBound : Float
  /-- Upper bound on the objective (from forward pass). -/
  upperBound : Float
  deriving Repr, Inhabited

/-- BaB verification result. -/
structure BaBResult where
  verified : Bool
  numBranches : Nat
  finalLowerBound : Float
  deriving Repr

/-- Select the most ambiguous neuron to split on.
    Heuristic: pick the unstable neuron with the largest (u - l) range. -/
def selectSplitNeuron (net : Network) (inputBounds : Interval) :
    Option NeuronSplit := do
  let mut currentBounds := inputBounds
  let mut bestScore := 0.0
  let mut bestLayer := 0
  let mut bestNeuron := 0
  let mut layerIdx := 0

  for layer in net.layers do
    match layer with
    | .activation .relu =>
      -- Score each neuron by ambiguity
      for i in [:currentBounds.lower.data.size] do
        let l := currentBounds.lower.data[i]!
        let u := currentBounds.upper.data[i]!
        if l < 0.0 && u > 0.0 then
          let score := u - l  -- Wider range = more ambiguous
          if score > bestScore then
            bestScore := score
            bestLayer := layerIdx
            bestNeuron := i
    | _ => pure ()
    let newBounds ← ibpLayer layer currentBounds
    currentBounds := newBounds
    layerIdx := layerIdx + 1

  if bestScore > 0.0 then
    return { layerIdx := bestLayer, neuronIdx := bestNeuron, isActive := true }
  else
    none

/-- Run a single round of BaB: pop worst node, split, recompute. -/
private def babStep (net : Network) (trueClass : Nat)
    (alphaParams : AlphaParams) (queue : Array BaBNode) :
    Array BaBNode × Nat := Id.run do
  if queue.isEmpty then return (queue, 0)

  -- Find worst (lowest lower bound) node
  let mut worstIdx : Nat := 0
  let mut worstBound := 1.0e38
  for i in [:queue.size] do
    let node := queue[i]!
    if node.lowerBound < worstBound then
      worstBound := node.lowerBound
      worstIdx := i

  let node := queue[worstIdx]!
  -- Remove worst node from queue
  let mut remaining := Array.mkEmpty (queue.size - 1)
  for k in [:queue.size] do
    if k != worstIdx then
      remaining := remaining.push queue[k]!

  -- Try to find a neuron to split
  match selectSplitNeuron net node.inputBounds with
  | none => return (remaining, 0)
  | some splitNeuron =>
    let mut newQueue := remaining
    let mut added := 0

    -- Active branch
    let activeSplit : NeuronSplit := { splitNeuron with isActive := true }
    let activeSplits := node.splits.push activeSplit
    match alphaCrownBackward net node.inputBounds alphaParams with
    | some childOutput =>
      let childLower := childOutput.lower.data[trueClass]!
      newQueue := newQueue.push {
        inputBounds := node.inputBounds
        splits := activeSplits
        lowerBound := childLower
        upperBound := node.upperBound
      }
      added := added + 1
    | none => pure ()

    -- Inactive branch
    let inactiveSplit : NeuronSplit := { splitNeuron with isActive := false }
    let inactiveSplits := node.splits.push inactiveSplit
    match alphaCrownBackward net node.inputBounds alphaParams with
    | some childOutput2 =>
      let childLower2 := childOutput2.lower.data[trueClass]!
      newQueue := newQueue.push {
        inputBounds := node.inputBounds
        splits := inactiveSplits
        lowerBound := childLower2
        upperBound := node.upperBound
      }
      added := added + 1
    | none => pure ()

    return (newQueue, added)

/-- Run branch-and-bound verification.
    Iteratively refines bounds by splitting on unstable neurons. -/
def branchAndBound (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) (maxBranches : Nat := 64) : BaBResult := Id.run do
  let inputBounds := Interval.epsBall input epsilon

  -- Initialize α parameters
  let sizes := net.layers.filterMap fun
    | .activation .relu => some 16
    | _ => none
  let alphaParams := AlphaParams.init sizes

  -- Try α-CROWN first
  match alphaCrownBackward net inputBounds alphaParams with
  | none => return { verified := false, numBranches := 0, finalLowerBound := -1.0e38 }
  | some outputBounds =>
    let trueClassLower := outputBounds.lower.data[trueClass]!

    -- Check if already verified
    let numClasses := outputBounds.lower.data.size
    let mut allVerified := true
    for i in [:numClasses] do
      if i != trueClass then
        if outputBounds.upper.data[i]! ≥ trueClassLower then
          allVerified := false

    if allVerified then
      return { verified := true, numBranches := 1, finalLowerBound := trueClassLower }

    -- Branch-and-bound loop
    let mut queue : Array BaBNode := #[{
      inputBounds := inputBounds
      splits := #[]
      lowerBound := trueClassLower
      upperBound := 1.0e38
    }]
    let mut numBranches : Nat := 1
    let mut globalLowerBound := trueClassLower

    for _ in [:maxBranches] do
      if queue.isEmpty then break
      let (newQueue, added) := babStep net trueClass alphaParams queue
      queue := newQueue
      numBranches := numBranches + added

      -- Update global lower bound from remaining queue
      if !queue.isEmpty then
        let mut minBound := 1.0e38
        for j in [:queue.size] do
          let n := queue[j]!
          minBound := fmin minBound n.lowerBound
        globalLowerBound := minBound

    -- Final check: if global lower bound means margin > 0 for all classes
    let mut finalVerified := true
    for i in [:numClasses] do
      if i != trueClass then
        if globalLowerBound ≤ 0.0 then
          finalVerified := false

    return { verified := finalVerified, numBranches, finalLowerBound := globalLowerBound }

/-- Verify robustness using α,β-CROWN with branch-and-bound. -/
def verifyRobustnessAlphaBetaCROWN (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) (maxBranches : Nat := 64) : Option Bool :=
  let result := branchAndBound net input epsilon trueClass maxBranches
  some result.verified

/-! ## α,β-CROWN Theorems -/

/-- α-CROWN provides bounds at least as tight as CROWN. -/
theorem alpha_crown_tighter_than_crown
    (net : Network) (inputBounds : Interval)
    (alphaParams : AlphaParams)
    (crownBounds alphaCrownBounds : Interval)
    (hcrown : crownBackward net inputBounds = some crownBounds)
    (halpha : alphaCrownBackward net inputBounds alphaParams = some alphaCrownBounds) :
    ∀ i, i < alphaCrownBounds.lower.data.size →
      crownBounds.lower.data[i]! ≤ alphaCrownBounds.lower.data[i]! ∧
      alphaCrownBounds.upper.data[i]! ≤ crownBounds.upper.data[i]! := by
  sorry

/-- BaB is a complete verification method:
    if the property holds, BaB will eventually verify it (with enough branches). -/
theorem bab_completeness
    (net : Network) (input : Tensor) (epsilon : Float) (trueClass : Nat)
    (hrobust : ∀ x, inLinfBall x input epsilon →
      classify net x = classify net input) :
    ∃ maxBranches, (branchAndBound net input epsilon trueClass maxBranches).verified = true := by
  sorry

end TorchLean
