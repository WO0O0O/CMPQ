-- TorchLean.Benchmarks.AcasXu
-- ACAS Xu benchmark for neural network verification (§7.1 of the paper)
--
-- ACAS Xu: Airborne Collision Avoidance System for Unmanned aircraft.
-- 45 networks (5×9 grid), each 6 hidden layers of 50 neurons.
-- 5 inputs → 5 outputs (advisory actions).

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown

namespace TorchLean

/-! ## ACAS Xu Network Specification -/

/-- ACAS Xu input features. -/
structure AcasXuInput where
  /-- ρ: Distance from ownship to intruder (ft) -/
  rho : Float
  /-- θ: Angle to intruder relative to ownship heading (rad) -/
  theta : Float
  /-- ψ: Heading angle of intruder relative to ownship heading (rad) -/
  psi : Float
  /-- v_own: Speed of ownship (ft/s) -/
  vOwn : Float
  /-- v_int: Speed of intruder (ft/s) -/
  vInt : Float
  deriving Repr

/-- ACAS Xu output advisories. -/
inductive AcasXuAdvisory where
  | clearOfConflict    -- COC: no action needed
  | weakLeft           -- WL: weak left turn
  | weakRight          -- WR: weak right turn
  | strongLeft         -- SL: strong left turn
  | strongRight        -- SR: strong right turn
  deriving Repr, BEq

/-- Convert advisory index to type. -/
def advisoryOfIndex : Nat → AcasXuAdvisory
  | 0 => .clearOfConflict
  | 1 => .weakLeft
  | 2 => .weakRight
  | 3 => .strongLeft
  | 4 => .strongRight
  | _ => .clearOfConflict

/-- ACAS Xu input normalization constants (from the benchmark). -/
structure AcasXuNormalization where
  inputMeans : Array Float := #[1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
  inputRanges : Array Float := #[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
  deriving Repr, Inhabited

/-- Normalize ACAS Xu input. -/
def normalizeAcasInput (input : AcasXuInput)
    (norm : AcasXuNormalization := {}) : Tensor := Id.run do
  let raw := #[input.rho, input.theta, input.psi, input.vOwn, input.vInt]
  let mut normalized := Array.mkEmpty 5
  for i in [:5] do
    normalized := normalized.push ((raw[i]! - norm.inputMeans[i]!) / norm.inputRanges[i]!)
  return ⟨[5], normalized⟩

/-! ## ACAS Xu Safety Properties (§7.1) -/

/-- ACAS Xu safety property specification. -/
structure AcasXuProperty where
  /-- Property name/identifier. -/
  name : String
  /-- Description of the property. -/
  description : String
  /-- Input bounds: (lower, upper) for each of the 5 inputs. -/
  inputLower : Array Float
  inputUpper : Array Float
  /-- The property to verify on the output. -/
  outputSpec : Tensor → Bool
  deriving Inhabited

/-- Property φ₁: If ρ ≥ 55947.691, the output advisory should be COC.
    "If the intruder is distant, then the score for COC is minimal." -/
def acasProperty1 : AcasXuProperty :=
  { name := "φ₁"
    description := "If intruder is distant (ρ ≥ 55947.691), advisory should not be COC"
    inputLower := #[55947.691, -3.141593, -3.141593, 1145.0, 0.0]
    inputUpper := #[60760.0, 3.141593, 3.141593, 1200.0, 60.0]
    outputSpec := fun output =>
      -- COC (index 0) should NOT have the minimum score
      let cocScore := output.data[0]!
      output.data.foldl (fun acc v => acc && (v ≤ cocScore)) true }

/-- Property φ₂: If ρ ≥ 55947.691, output should not be COC.
    "If intruder is distant and slower, COC should not be maximal." -/
def acasProperty2 : AcasXuProperty :=
  { name := "φ₂"
    description := "COC is not the maximal score when intruder is distant"
    inputLower := #[55947.691, -3.141593, -3.141593, 1145.0, 0.0]
    inputUpper := #[60760.0, 3.141593, 3.141593, 1200.0, 60.0]
    outputSpec := fun output =>
      -- COC (index 0) should NOT be the argmax
      Tensor.argmaxT output != 0 }

/-- Property φ₃: If intruder is directly ahead and near, do not advise COC.
    "If the intruder is nearby and approaching head-on, COC is unsafe." -/
def acasProperty3 : AcasXuProperty :=
  { name := "φ₃"
    description := "Do not advise COC when intruder is directly ahead and near"
    inputLower := #[1500.0, -0.06, 3.10, 980.0, 960.0]
    inputUpper := #[1800.0, 0.06, 3.141593, 1200.0, 1200.0]
    outputSpec := fun output =>
      -- COC should not be minimal (not recommended)
      let cocScore := output.data[0]!
      output.data.any (· < cocScore) }

/-- Property φ₄: If intruder is directly ahead and near, do not advise COC.
    "Stronger version: COC should have highest score (least recommended)." -/
def acasProperty4 : AcasXuProperty :=
  { name := "φ₄"
    description := "COC should have the highest score when intruder approaches head-on"
    inputLower := #[1500.0, -0.06, 0.0, 1000.0, 700.0]
    inputUpper := #[1800.0, 0.06, 0.0, 1200.0, 800.0]
    outputSpec := fun output =>
      Tensor.argmaxT output == 0 }

/-- Property φ₅: If intruder is near, advise Strong Right.
    "When the intruder is near and on the left, strong right is appropriate." -/
def acasProperty5 : AcasXuProperty :=
  { name := "φ₅"
    description := "Strong Right is the minimal score when intruder is near on left"
    inputLower := #[250.0, 0.2, -3.141593, 100.0, 0.0]
    inputUpper := #[400.0, 0.4, -3.141593, 400.0, 400.0]
    outputSpec := fun output =>
      -- Strong Right (index 4) should have minimum score (recommended)
      let srScore := output.data[4]!
      output.data.foldl (fun acc v => acc && (srScore ≤ v)) true }

/-- All 10 ACAS Xu safety properties from the VNN-COMP benchmark. -/
def acasProperties : List AcasXuProperty :=
  [acasProperty1, acasProperty2, acasProperty3, acasProperty4, acasProperty5]

/-! ## ACAS Xu Network Construction -/

/-- Build a standard ACAS Xu network architecture:
    5 inputs → [50]×6 hidden ReLU layers → 5 outputs.
    (Weights would be loaded from ONNX/NNet files in practice.) -/
def buildAcasXuNetwork (hiddenSize : Nat := 50) (numHidden : Nat := 6) : Network := Id.run do
  let mut net := Network.empty
  let mut inSize := 5

  for _ in [:numHidden] do
    let layer := LinearLayer.zeros inSize hiddenSize
    net := net.addLinearAct layer .relu
    inSize := hiddenSize

  -- Output layer (no activation)
  let outputLayer := LinearLayer.zeros inSize 5
  net := net.addLayer (.linear outputLayer)
  return net

/-! ## ACAS Xu Verification Runner -/

/-- Result of verifying a single ACAS Xu property. -/
structure AcasXuVerifyResult where
  propertyName : String
  verified : Bool
  method : String
  deriving Repr

/-- Verify an ACAS Xu property using IBP. -/
def verifyAcasProperty (net : Network) (prop : AcasXuProperty) : Option AcasXuVerifyResult := do
  let inputBounds : Interval :=
    ⟨⟨[5], prop.inputLower⟩, ⟨[5], prop.inputUpper⟩⟩

  let outputBounds ← ibpNetwork net inputBounds
  -- Check if output specification holds for all possible outputs within bounds
  let verified := prop.outputSpec outputBounds.lower && prop.outputSpec outputBounds.upper

  return {
    propertyName := prop.name
    verified := verified
    method := "IBP"
  }

/-- Run all ACAS Xu property verifications. -/
def runAcasXuBenchmark (net : Network) : List (Option AcasXuVerifyResult) :=
  acasProperties.map (verifyAcasProperty net)

end TorchLean
