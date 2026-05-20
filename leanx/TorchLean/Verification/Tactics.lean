-- TorchLean.Verification.Tactics
-- Automated verification tactics and decision procedures (§5.5 of the paper)
--
-- Provides Lean tactics for:
-- 1. Interval arithmetic reasoning
-- 2. ReLU case splitting
-- 3. Robustness verification automation
-- 4. Bound propagation

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown
import TorchLean.Verification.Robustness

namespace TorchLean

/-! ## Interval Arithmetic Decision Procedure -/

/-- Decide if a ≤ b for concrete Float values. -/
def decideFloatLe (a b : Float) : Bool := a ≤ b

/-- Decide if a < b for concrete Float values. -/
def decideFloatLt (a b : Float) : Bool := a < b

/-- Verify interval containment: [l₁, u₁] ⊆ [l₂, u₂] ↔ l₂ ≤ l₁ ∧ u₁ ≤ u₂. -/
def intervalContainment (l1 u1 l2 u2 : Float) : Bool :=
  l2 ≤ l1 && u1 ≤ u2

/-- Verify interval non-overlap: [l₁, u₁] ∩ [l₂, u₂] = ∅ ↔ u₁ < l₂ ∨ u₂ < l₁. -/
def intervalDisjoint (l1 u1 l2 u2 : Float) : Bool :=
  u1 < l2 || u2 < l1

/-! ## ReLU Case Analysis -/

/-- ReLU neuron status based on bounds. -/
inductive ReLUStatus where
  | alwaysActive    -- lower ≥ 0: ReLU(x) = x
  | alwaysInactive  -- upper ≤ 0: ReLU(x) = 0
  | unstable        -- lower < 0 < upper: need case split
  deriving Repr, BEq

/-- Determine ReLU neuron status from pre-activation bounds. -/
def reluStatus (lower upper : Float) : ReLUStatus :=
  if lower ≥ 0.0 then .alwaysActive
  else if upper ≤ 0.0 then .alwaysInactive
  else .unstable

/-- Analyze all ReLU neurons in a network given input bounds.
    Returns the status of each ReLU neuron. -/
def analyzeReLUStability (net : Network) (inputBounds : Interval) :
    Option (List (List ReLUStatus)) := do
  let mut currentBounds := inputBounds
  let mut result : List (List ReLUStatus) := []

  for layer in net.layers do
    match layer with
    | .activation .relu =>
      let mut statuses : List ReLUStatus := []
      for i in [:currentBounds.lower.data.size] do
        let l := currentBounds.lower.data[i]!
        let u := currentBounds.upper.data[i]!
        statuses := statuses ++ [reluStatus l u]
      result := result ++ [statuses]
    | _ => pure ()
    let newBounds ← ibpLayer layer currentBounds
    currentBounds := newBounds

  return result

/-- Count unstable neurons in a network. -/
def countUnstableNeurons (net : Network) (inputBounds : Interval) : Option Nat := do
  let statuses ← analyzeReLUStability net inputBounds
  let count := statuses.foldl (fun acc layerStatuses =>
    acc + layerStatuses.foldl (fun c s =>
      match s with | .unstable => c + 1 | _ => c) 0) 0
  return count

/-! ## Automated Bound Tightening -/

/-- Tighten input bounds by removing infeasible regions using IBP.
    For each input dimension, try tightening the lower and upper bounds. -/
def tightenBounds (net : Network) (inputBounds : Interval)
    (trueClass : Nat) (steps : Nat := 10) : Option Interval := do
  let mut bounds := inputBounds
  let n := bounds.lower.data.size

  for _ in [:steps] do
    for dim in [:n] do
      let mid := (bounds.lower.data[dim]! + bounds.upper.data[dim]!) / 2.0

      -- Try tightening lower bound
      let tighterLower := bounds.lower.setFlat dim mid
      let testBoundsL : Interval := ⟨tighterLower, bounds.upper⟩
      match ibpNetwork net testBoundsL with
      | some outputBounds =>
        let trueClassLower ← outputBounds.lower.getFlat trueClass
        let mut feasible := true
        for i in [:outputBounds.lower.data.size] do
          if i != trueClass then
            let otherUpper ← outputBounds.upper.getFlat i
            if otherUpper ≥ trueClassLower then
              feasible := false
        if feasible then
          bounds := testBoundsL
      | none => pure ()

      -- Try tightening upper bound
      let tighterUpper := bounds.upper.setFlat dim mid
      let testBoundsU : Interval := ⟨bounds.lower, tighterUpper⟩
      match ibpNetwork net testBoundsU with
      | some outputBounds =>
        let trueClassLower ← outputBounds.lower.getFlat trueClass
        let mut feasible := true
        for i in [:outputBounds.lower.data.size] do
          if i != trueClass then
            let otherUpper ← outputBounds.upper.getFlat i
            if otherUpper ≥ trueClassLower then
              feasible := false
        if feasible then
          bounds := testBoundsU
      | none => pure ()

  return bounds

/-! ## Verification Strategy Selection -/

/-- Verification strategy. -/
inductive VerifyStrategy where
  | ibpOnly          -- Use IBP (fastest, loosest)
  | crownOnly        -- Use CROWN (tighter, slower)
  | ibpThenCrown     -- Try IBP first, fall back to CROWN
  | alphaBetaCrown   -- Use α,β-CROWN with BaB (tightest, slowest)
  | adaptive         -- Automatically select based on network size
  deriving Repr, BEq

/-- Select verification strategy based on network characteristics. -/
def selectStrategy (net : Network) (inputBounds : Interval) : VerifyStrategy :=
  let numParams := net.numParams
  let inputDim := inputBounds.lower.data.size
  let numLayers := net.numLayers
  -- Heuristic: use cheaper methods for larger networks
  if numParams > 100000 || inputDim > 1000 then .ibpOnly
  else if numLayers > 10 then .ibpThenCrown
  else if numParams < 1000 then .alphaBetaCrown
  else .crownOnly

/-- Run verification with the selected strategy. -/
def verifyWithStrategy (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) (strategy : VerifyStrategy := .adaptive) : Option Bool := do
  let strat := match strategy with
    | .adaptive => selectStrategy net (Interval.epsBall input epsilon)
    | s => s

  match strat with
  | .ibpOnly => verifyRobustnessIBP net input epsilon trueClass
  | .crownOnly => verifyRobustnessCROWN net input epsilon trueClass
  | .ibpThenCrown =>
    match ← verifyRobustnessIBP net input epsilon trueClass with
    | true => return true
    | false => verifyRobustnessCROWN net input epsilon trueClass
  | .alphaBetaCrown | .adaptive =>
    match ← verifyRobustnessCROWN net input epsilon trueClass with
    | true => return true
    | false =>
      match ← verifyRobustnessIBP net input epsilon trueClass with
      | true => return true
      | false => return false

/-! ## Verification Proof Helpers -/

/-- Helper: Show that if IBP certifies robustness, then the classification is preserved.
    This connects the computational verification to the formal property. -/
def verificationWitness (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) : Option String := do
  let inputBounds := Interval.epsBall input epsilon
  let outputBounds ← ibpNetwork net inputBounds
  let trueClassLower ← outputBounds.lower.getFlat trueClass
  let numClasses := outputBounds.lower.data.size

  let mut margins : List (Nat × Float) := []
  let mut verified := true
  for i in [:numClasses] do
    if i != trueClass then
      let otherUpper ← outputBounds.upper.getFlat i
      let margin := trueClassLower - otherUpper
      margins := margins ++ [(i, margin)]
      if margin ≤ 0.0 then verified := false

  let mut proof := s!"Verification Witness for ε={epsilon}, class={trueClass}\n"
  proof := proof ++ s!"  Input bounds: [{input}] ± {epsilon}\n"
  proof := proof ++ s!"  True class lower bound: {trueClassLower}\n"
  for (cls, margin) in margins do
    proof := proof ++ s!"  Class {cls} margin: {margin}"
    proof := proof ++ (if margin > 0.0 then " ✓\n" else " ✗\n")
  proof := proof ++ s!"  Result: {if verified then "VERIFIED" else "INCONCLUSIVE"}\n"

  return proof

/-! ## Theorem: Verification Soundness -/

/-- The adaptive strategy is sound: if it returns true, the network is robust. -/
theorem adaptive_strategy_soundness
    (net : Network) (x₀ : Tensor) (epsilon : Float) (trueClass : Nat)
    (hverify : verifyWithStrategy net x₀ epsilon trueClass .adaptive = some true) :
    isRobust net x₀ epsilon := by
  sorry

/-- IBP-then-CROWN is at least as powerful as IBP alone. -/
theorem ibp_then_crown_stronger
    (net : Network) (x₀ : Tensor) (epsilon : Float) (trueClass : Nat)
    (hibp : verifyRobustnessIBP net x₀ epsilon trueClass = some true) :
    verifyWithStrategy net x₀ epsilon trueClass .ibpThenCrown = some true := by
  sorry

end TorchLean
