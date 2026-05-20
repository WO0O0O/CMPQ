-- TorchLean: End-to-end demonstration
-- Formalizing Neural Networks in Lean 4

import TorchLean

open TorchLean

/-- Build a simple 2-layer ReLU network for binary classification.
    Architecture: 2 → 4 → ReLU → 2 -/
def buildDemoNetwork : Option Network := do
  -- Layer 1: Linear 2 → 4
  let w1 ← Tensor.ofData [4, 2] #[
     0.5,  0.3,
    -0.2,  0.8,
     0.7, -0.4,
     0.1,  0.6
  ]
  let b1 ← Tensor.ofData [4] #[0.1, -0.1, 0.2, 0.0]
  let linear1 : LinearLayer := ⟨2, 4, w1, b1⟩

  -- Layer 2: Linear 4 → 2
  let w2 ← Tensor.ofData [2, 4] #[
     0.4, -0.3,  0.6,  0.2,
    -0.5,  0.7, -0.1,  0.3
  ]
  let b2 ← Tensor.ofData [2] #[0.1, -0.1]
  let linear2 : LinearLayer := ⟨4, 2, w2, b2⟩

  let net := Network.empty
    |>.addLinearAct linear1 .relu
    |>.addLayer (.linear linear2)
  return net

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║  TorchLean: Neural Network Verification  ║"
  IO.println "╚══════════════════════════════════════════╝"
  IO.println ""

  -- 1. Build network
  IO.println "── 1. Network Construction ──"
  let some net := buildDemoNetwork | IO.println "ERROR: Failed to build network"; return
  IO.println s!"  Layers: {net.numLayers}"
  IO.println s!"  Parameters: {net.numParams}"
  IO.println ""

  -- 2. Forward pass
  IO.println "── 2. Forward Pass ──"
  let some input := Tensor.ofData [2] #[0.5, 0.8] | return
  IO.println s!"  Input:  {input}"
  let some output := net.forward input | IO.println "ERROR: Forward pass failed"; return
  IO.println s!"  Output: {output}"
  let predicted := Tensor.argmaxT output
  IO.println s!"  Predicted class: {predicted}"
  IO.println ""

  -- 3. Computation graph compilation
  IO.println "── 3. Computation Graph ──"
  let graph := ComputeGraph.ofNetwork net [2]
  IO.println s!"{graph}"

  -- Verify graph produces same result
  let some graphOutput := graph.eval [input] | IO.println "ERROR: Graph eval failed"; return
  IO.println s!"  Graph output: {graphOutput}"
  IO.println ""

  -- 4. IBP verification
  IO.println "── 4. IBP Robustness Verification ──"
  let epsilon := 0.05
  IO.println s!"  ε = {epsilon}"

  let some ibpResult := verifyRobustnessIBP net input epsilon predicted
    | IO.println "ERROR: IBP failed"; return
  IO.println s!"  IBP robust: {ibpResult}"

  -- Show IBP bounds
  let ibpBounds := Interval.epsBall input epsilon
  let some ibpOutput := ibpNetwork net ibpBounds | return
  IO.println s!"  IBP output lower: {ibpOutput.lower}"
  IO.println s!"  IBP output upper: {ibpOutput.upper}"
  IO.println ""

  -- 5. CROWN verification
  IO.println "── 5. CROWN Robustness Verification ──"
  let some crownResult := verifyRobustnessCROWN net input epsilon predicted
    | IO.println "ERROR: CROWN failed"; return
  IO.println s!"  CROWN robust: {crownResult}"

  let some crownOutput := crownBackward net ibpBounds | return
  IO.println s!"  CROWN output lower: {crownOutput.lower}"
  IO.println s!"  CROWN output upper: {crownOutput.upper}"
  IO.println ""

  -- 6. α,β-CROWN verification
  IO.println "── 6. α,β-CROWN Verification ──"
  let babResult := branchAndBound net input epsilon predicted 32
  IO.println s!"  α,β-CROWN verified: {babResult.verified}"
  IO.println s!"  Branches explored: {babResult.numBranches}"
  IO.println s!"  Final lower bound: {babResult.finalLowerBound}"
  IO.println ""

  -- 7. Certificate generation & verification
  IO.println "── 7. Certificate Generation ──"
  let some certIBP := Certificate.generate net input epsilon predicted .ibp | return
  IO.println s!"  {certIBP}"
  IO.println ""

  let some verified := certIBP.verify net | return
  IO.println s!"  Certificate verified: {verified}"
  IO.println ""

  -- 8. Compare methods
  IO.println "── 8. Method Comparison ──"
  let some (ibpW, crownW) := compareMethods net input epsilon | return
  IO.println s!"  IBP total output width:   {ibpW}"
  IO.println s!"  CROWN total output width: {crownW}"
  if crownW < ibpW then
    IO.println "  → CROWN provides tighter bounds"
  else
    IO.println "  → IBP and CROWN bounds are comparable"
  IO.println ""

  -- 9. Adversarial attacks
  IO.println "── 9. Adversarial Attacks ──"
  let some fgsmAdv := fgsm net input predicted epsilon | IO.println "FGSM failed"; return
  let some fgsmResult := evaluateAttack net input fgsmAdv predicted | return
  IO.println s!"  FGSM: successful={fgsmResult.isSuccessful}, perturbation={fgsmResult.perturbationNorm}"
  IO.println s!"    Original class: {fgsmResult.originalClass}, Adversarial class: {fgsmResult.adversarialClass}"

  let some pgdAdv := pgd net input predicted epsilon | IO.println "PGD failed"; return
  let some pgdResult := evaluateAttack net input pgdAdv predicted | return
  IO.println s!"  PGD:  successful={pgdResult.isSuccessful}, perturbation={pgdResult.perturbationNorm}"
  IO.println s!"    Original class: {pgdResult.originalClass}, Adversarial class: {pgdResult.adversarialClass}"
  IO.println ""

  -- 10. ReLU stability analysis
  IO.println "── 10. ReLU Stability Analysis ──"
  let some unstableCount := countUnstableNeurons net ibpBounds | return
  IO.println s!"  Unstable ReLU neurons (ε={epsilon}): {unstableCount}"
  let some statuses := analyzeReLUStability net ibpBounds | return
  for (layerStatuses, idx) in statuses.zip (List.range statuses.length) do
    let active := layerStatuses.filter (· == .alwaysActive) |>.length
    let inactive := layerStatuses.filter (· == .alwaysInactive) |>.length
    let unstable := layerStatuses.filter (· == .unstable) |>.length
    IO.println s!"  Layer {idx}: active={active}, inactive={inactive}, unstable={unstable}"
  IO.println ""

  -- 11. Verification strategy selection
  IO.println "── 11. Adaptive Verification ──"
  let strategy := selectStrategy net ibpBounds
  IO.println s!"  Selected strategy: {repr strategy}"
  let some adaptiveResult := verifyWithStrategy net input epsilon predicted .adaptive | return
  IO.println s!"  Adaptive result: {adaptiveResult}"
  IO.println ""

  -- 12. Float32 formalization demo
  IO.println "── 12. IEEE-754 Float32 Formalization ──"
  let f1 := Float32.posZero
  IO.println s!"  +0: category = {repr (f1.category)}"
  let f2 := Float32.posInf
  IO.println s!"  +∞: category = {repr (f2.category)}, isInf = {f2.isInf}"
  IO.println s!"  Machine epsilon: {Float32.machineEpsilon}"
  IO.println ""

  -- 13. ACAS Xu benchmark demo
  IO.println "── 13. ACAS Xu Benchmark ──"
  let acasNet := buildAcasXuNetwork
  IO.println s!"  ACAS Xu network: {acasNet.numLayers} layers, {acasNet.numParams} params"
  IO.println s!"  Properties defined: {acasProperties.length}"
  for prop in acasProperties do
    IO.println s!"    {prop.name}: {prop.description}"
  IO.println ""

  -- 14. MNIST benchmark demo
  IO.println "── 14. MNIST/CIFAR-10 Benchmark ──"
  let mnistNet := buildMNISTSmall
  IO.println s!"  MNIST-Small: {mnistNet.numLayers} layers, {mnistNet.numParams} params"
  let cifar10Net := buildCIFAR10Small
  IO.println s!"  CIFAR-10-Small: {cifar10Net.numLayers} layers, {cifar10Net.numParams} params"
  IO.println s!"  MNIST epsilons: {mnistEpsilons}"
  IO.println s!"  CIFAR-10 epsilons: {cifar10Epsilons}"
  IO.println ""

  -- 15. Verification witness
  IO.println "── 15. Verification Witness ──"
  let some witness := verificationWitness net input epsilon predicted | return
  IO.println witness

  IO.println "Done."
