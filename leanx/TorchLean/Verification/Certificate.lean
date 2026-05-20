-- TorchLean.Verification.Certificate
-- Verification certificate generation and checking

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown
import TorchLean.Verification.AlphaBetaCrown

namespace TorchLean

/-- Verification method used to generate the certificate. -/
inductive VerificationMethod where
  | ibp           -- Interval Bound Propagation
  | crown         -- CROWN/LiRPA
  | alphaBetaCrown -- α,β-CROWN with branch-and-bound
  deriving Repr, BEq

/-- A verification certificate asserting robustness of a network output. -/
structure Certificate where
  method : VerificationMethod
  inputCenter : Tensor        -- Center of input perturbation region
  epsilon : Float             -- L∞ perturbation bound
  trueClass : Nat             -- Expected classification
  outputLower : Tensor        -- Lower bounds on output
  outputUpper : Tensor        -- Upper bounds on output
  isRobust : Bool             -- Whether robustness was verified
  deriving Repr

namespace Certificate

/-- Generate a robustness certificate using the specified method. -/
def generate (net : Network) (input : Tensor) (epsilon : Float)
    (trueClass : Nat) (method : VerificationMethod := .ibp) : Option Certificate := do
  let inputBounds := Interval.epsBall input epsilon

  let outputBounds ← match method with
    | .ibp   => ibpNetwork net inputBounds
    | .crown => crownBackward net inputBounds
    | .alphaBetaCrown =>
      let alphaParams := AlphaParams.init []
      alphaCrownBackward net inputBounds alphaParams

  -- Check robustness
  let trueClassLower ← outputBounds.lower.getFlat trueClass
  let numClasses := outputBounds.lower.data.size
  let mut robust := true
  for i in [:numClasses] do
    if i != trueClass then
      let otherUpper ← outputBounds.upper.getFlat i
      if otherUpper ≥ trueClassLower then
        robust := false

  return {
    method := method
    inputCenter := input
    epsilon := epsilon
    trueClass := trueClass
    outputLower := outputBounds.lower
    outputUpper := outputBounds.upper
    isRobust := robust
  }

/-- Verify a certificate by independently recomputing bounds and checking. -/
def verify (cert : Certificate) (net : Network) : Option Bool := do
  let inputBounds := Interval.epsBall cert.inputCenter cert.epsilon
  let outputBounds ← match cert.method with
    | .ibp   => ibpNetwork net inputBounds
    | .crown => crownBackward net inputBounds
    | .alphaBetaCrown =>
      let alphaParams := AlphaParams.init []
      alphaCrownBackward net inputBounds alphaParams

  -- Verify that certificate bounds are valid (at least as tight)
  let numOutputs := cert.outputLower.data.size
  for i in [:numOutputs] do
    let certLower ← cert.outputLower.getFlat i
    let recomputedLower ← outputBounds.lower.getFlat i
    let certUpper ← cert.outputUpper.getFlat i
    let recomputedUpper ← outputBounds.upper.getFlat i
    -- Certificate lower bound should not exceed recomputed lower bound
    if certLower > recomputedLower + 1e-6 then return false
    -- Certificate upper bound should not be below recomputed upper bound
    if certUpper < recomputedUpper - 1e-6 then return false

  -- Verify robustness claim
  if cert.isRobust then
    let trueClassLower ← outputBounds.lower.getFlat cert.trueClass
    for i in [:numOutputs] do
      if i != cert.trueClass then
        let otherUpper ← outputBounds.upper.getFlat i
        if otherUpper ≥ trueClassLower then return false

  return true

/-- Pretty-print a certificate. -/
def toString (cert : Certificate) : String :=
  let methodStr := match cert.method with
    | .ibp   => "IBP"
    | .crown => "CROWN"
    | .alphaBetaCrown => "α,β-CROWN"
  let robustStr := if cert.isRobust then "VERIFIED ROBUST" else "NOT VERIFIED"
  s!"Certificate ({methodStr}):\n" ++
  s!"  epsilon: {cert.epsilon}\n" ++
  s!"  true class: {cert.trueClass}\n" ++
  s!"  output lower: {cert.outputLower}\n" ++
  s!"  output upper: {cert.outputUpper}\n" ++
  s!"  result: {robustStr}"

instance : ToString Certificate := ⟨Certificate.toString⟩

end Certificate

/-! ## Batch verification -/

/-- Verify robustness for a batch of inputs. Returns the fraction verified. -/
def batchVerify (net : Network) (inputs : List Tensor) (epsilon : Float)
    (trueClasses : List Nat) (method : VerificationMethod := .ibp) : Option Float := do
  if inputs.length != trueClasses.length then return 0.0
  let mut verified := 0
  let mut total := 0
  for (input, trueClass) in inputs.zip trueClasses do
    total := total + 1
    let cert ← Certificate.generate net input epsilon trueClass method
    if cert.isRobust then
      verified := verified + 1
  return Float.ofNat verified / Float.ofNat total

/-! ## Comparison of verification methods -/

/-- Compare IBP and CROWN bounds for the same input.
    Returns (ibpWidth, crownWidth) where width = upper - lower. -/
def compareMethods (net : Network) (input : Tensor) (epsilon : Float) :
    Option (Float × Float) := do
  let inputBounds := Interval.epsBall input epsilon

  let ibpBounds ← ibpNetwork net inputBounds
  let crownBounds ← crownBackward net inputBounds

  let ibpWidthTensor ← Tensor.sub ibpBounds.upper ibpBounds.lower
  let crownWidthTensor ← Tensor.sub crownBounds.upper crownBounds.lower

  return (ibpWidthTensor.sum, crownWidthTensor.sum)

end TorchLean
