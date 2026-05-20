-- TorchLean.Verification.Attacks
-- Adversarial attack implementations (§5.4 of the paper)
--
-- FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent)
-- Used for finding adversarial examples and testing robustness.

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP

namespace TorchLean

/-! ## Loss Functions -/

/-- Cross-entropy loss for classification (negative log-likelihood).
    Computes -log(softmax(output)[trueClass]). -/
def crossEntropyLoss (output : Tensor) (trueClass : Nat) : Option Float := do
  -- Compute max for numerical stability
  let maxVal ← output.maxElem
  let shifted := Tensor.map (· - maxVal) output
  let expSum := shifted.data.foldl (fun acc v => acc + Float.exp v) 0.0
  let logSumExp := Float.log expSum + maxVal
  let trueLogit ← output.getFlat trueClass
  return logSumExp - trueLogit

/-- Margin loss: maximize the gap between true class and best other class.
    loss = max_{i ≠ trueClass} output[i] - output[trueClass] -/
def marginLoss (output : Tensor) (trueClass : Nat) : Option Float := do
  let trueLogit ← output.getFlat trueClass
  let mut maxOther := -1.0e38
  for i in [:output.data.size] do
    if i != trueClass then
      let v := output.data[i]!
      maxOther := fmax maxOther v
  return maxOther - trueLogit

/-! ## Gradient Estimation -/

/-- Estimate gradient of loss w.r.t. input using finite differences.
    ∂L/∂xᵢ ≈ (L(x + h·eᵢ) - L(x - h·eᵢ)) / (2h) -/
def estimateGradient (net : Network) (input : Tensor) (trueClass : Nat)
    (h : Float := 1e-4) : Option Tensor := do
  let n := input.data.size
  let mut gradData := Array.mkEmpty n

  for i in [:n] do
    -- Forward pass at x + h·eᵢ
    let inputPlus := input.setFlat i (input.data[i]! + h)
    let outPlus ← net.forward inputPlus
    let lossPlus ← crossEntropyLoss outPlus trueClass

    -- Forward pass at x - h·eᵢ
    let inputMinus := input.setFlat i (input.data[i]! - h)
    let outMinus ← net.forward inputMinus
    let lossMinus ← crossEntropyLoss outMinus trueClass

    -- Central difference
    gradData := gradData.push ((lossPlus - lossMinus) / (2.0 * h))

  return ⟨input.shape, gradData⟩

/-! ## FGSM (Fast Gradient Sign Method) -/

/-- FGSM attack: x_adv = x + ε · sign(∇_x L(x, y))
    Single-step attack that moves in the gradient sign direction. -/
def fgsm (net : Network) (input : Tensor) (trueClass : Nat)
    (epsilon : Float) : Option Tensor := do
  -- Estimate gradient
  let grad ← estimateGradient net input trueClass

  -- Compute perturbation: ε · sign(grad)
  let perturbation := Tensor.map (fun g =>
    if g > 0.0 then epsilon
    else if g < 0.0 then -epsilon
    else 0.0) grad

  -- Apply perturbation
  Tensor.add input perturbation

/-- Targeted FGSM: minimize loss for target class instead. -/
def fgsmTargeted (net : Network) (input : Tensor) (targetClass : Nat)
    (epsilon : Float) : Option Tensor := do
  let grad ← estimateGradient net input targetClass
  -- Move in negative gradient direction to minimize target loss
  let perturbation := Tensor.map (fun g =>
    if g > 0.0 then -epsilon
    else if g < 0.0 then epsilon
    else 0.0) grad
  Tensor.add input perturbation

/-! ## PGD (Projected Gradient Descent) -/

/-- Project a tensor onto the L∞ ball: clip to [center - ε, center + ε]. -/
def projectLinf (x center : Tensor) (epsilon : Float) : Tensor := Id.run do
  let mut clippedData := Array.mkEmpty x.data.size
  for i in [:x.data.size] do
    let v := x.data[i]!
    if i < center.data.size then
      clippedData := clippedData.push (fclamp v (center.data[i]! - epsilon) (center.data[i]! + epsilon))
    else
      clippedData := clippedData.push v
  return ⟨x.shape, clippedData⟩

/-- PGD attack: iterative FGSM with projection.
    x_{t+1} = Π_{B(x₀, ε)} (x_t + α · sign(∇_x L(x_t, y)))
    where Π projects back onto the L∞ ball. -/
def pgd (net : Network) (input : Tensor) (trueClass : Nat)
    (epsilon : Float) (stepSize : Float := 0.0) (numSteps : Nat := 20)
    (numRestarts : Nat := 1) : Option Tensor := do
  let alpha := if stepSize > 0.0 then stepSize else epsilon / 4.0
  let mut bestAdv := input
  let mut bestLoss := -1.0e38

  for restart in [:numRestarts] do
    -- Initialize: random start within ε-ball (deterministic using restart index)
    let mut current := input
    if restart > 0 then
      -- Pseudo-random perturbation based on restart index
      let mut pertData := Array.mkEmpty input.data.size
      for i in [:input.data.size] do
        let seed := Float.ofNat (restart * 1000 + i)
        let pseudoRand := Float.sin (seed * 12.9898) * 43758.5453
        let r := pseudoRand - Float.floor pseudoRand  -- [0, 1)
        let pert := (2.0 * r - 1.0) * epsilon
        pertData := pertData.push (input.data[i]! + pert)
      current := projectLinf ⟨input.shape, pertData⟩ input epsilon

    -- PGD iterations
    for _ in [:numSteps] do
      -- Estimate gradient
      let grad ← estimateGradient net current trueClass

      -- Step in sign direction
      let perturbation := Tensor.map (fun g =>
        if g > 0.0 then alpha
        else if g < 0.0 then -alpha
        else 0.0) grad
      let stepped ← Tensor.add current perturbation

      -- Project back onto ε-ball
      current := projectLinf stepped input epsilon

    -- Evaluate attack success
    let output ← net.forward current
    let loss ← marginLoss output trueClass
    if loss > bestLoss then
      bestLoss := loss
      bestAdv := current

  return bestAdv

/-! ## Attack Evaluation -/

/-- Result of an adversarial attack attempt. -/
structure AttackResult where
  /-- The adversarial example (if found). -/
  adversarialInput : Tensor
  /-- Whether the attack changed the classification. -/
  isSuccessful : Bool
  /-- Original predicted class. -/
  originalClass : Nat
  /-- Adversarial predicted class. -/
  adversarialClass : Nat
  /-- L∞ distance of perturbation. -/
  perturbationNorm : Float
  deriving Repr

/-- Evaluate an adversarial example against the original input. -/
def evaluateAttack (net : Network) (original adversarial : Tensor)
    (trueClass : Nat) : Option AttackResult := do
  let origOut ← net.forward original
  let advOut ← net.forward adversarial
  let origClass := Tensor.argmaxT origOut
  let advClass := Tensor.argmaxT advOut

  -- Compute L∞ perturbation norm
  let diff ← Tensor.sub adversarial original
  let linfNorm := diff.data.foldl (fun acc v => fmax acc (fabs v)) 0.0

  return {
    adversarialInput := adversarial
    isSuccessful := advClass != trueClass
    originalClass := origClass
    adversarialClass := advClass
    perturbationNorm := linfNorm
  }

/-- Run both FGSM and PGD attacks and return results. -/
def runAttacks (net : Network) (input : Tensor) (trueClass : Nat)
    (epsilon : Float) : Option (AttackResult × AttackResult) := do
  -- FGSM attack
  let fgsmAdv ← fgsm net input trueClass epsilon
  let fgsmResult ← evaluateAttack net input fgsmAdv trueClass

  -- PGD attack
  let pgdAdv ← pgd net input trueClass epsilon
  let pgdResult ← evaluateAttack net input pgdAdv trueClass

  return (fgsmResult, pgdResult)

/-- Empirical robustness rate: fraction of inputs that resist attack. -/
def empiricalRobustness (net : Network) (inputs : List Tensor)
    (trueClasses : List Nat) (epsilon : Float) : Option Float := do
  if inputs.length != trueClasses.length then return 0.0
  let mut robust := 0
  let mut total := 0
  for (input, trueClass) in inputs.zip trueClasses do
    total := total + 1
    let pgdAdv ← pgd net input trueClass epsilon
    let result ← evaluateAttack net input pgdAdv trueClass
    if !result.isSuccessful then
      robust := robust + 1
  return Float.ofNat robust / Float.ofNat total

end TorchLean
