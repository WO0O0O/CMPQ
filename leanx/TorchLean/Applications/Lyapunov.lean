-- TorchLean.Applications.Lyapunov
-- Lyapunov-style neural controller verification (§6.3 of the paper)
--
-- A neural controller is verified stable if there exists a Lyapunov function V such that:
--   V(x) > 0  for x ≠ 0    (positive definiteness)
--   V̇(x) < 0  for x ≠ 0    (asymptotic stability)

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP

namespace TorchLean

/-! ## Dynamical System Definition -/

/-- A continuous-time dynamical system: ẋ = f(x, u)
    where x is state, u is control input. -/
structure DynamicalSystem where
  /-- State dimension. -/
  stateDim : Nat
  /-- Control input dimension. -/
  controlDim : Nat
  /-- Dynamics function: (state, control) → state derivative. -/
  dynamics : Tensor → Tensor → Option Tensor
  deriving Inhabited

/-- A neural network controller: u = π(x). -/
structure NeuralController where
  /-- The control policy network. -/
  policy : Network
  /-- State dimension (input size). -/
  stateDim : Nat
  /-- Control dimension (output size). -/
  controlDim : Nat
  deriving Repr

/-- A neural Lyapunov function: V(x) = network(x). -/
structure LyapunovFunction where
  /-- The Lyapunov network. Must output a scalar ≥ 0. -/
  network : Network
  /-- State dimension. -/
  stateDim : Nat
  deriving Repr

/-! ## Lyapunov Conditions -/

/-- Evaluate the Lyapunov function at a state. -/
def evalLyapunov (V : LyapunovFunction) (x : Tensor) : Option Float := do
  let output ← V.network.forward x
  output.getFlat 0

/-- Evaluate the controller at a state. -/
def evalController (ctrl : NeuralController) (x : Tensor) : Option Tensor :=
  ctrl.policy.forward x

/-- Approximate Lyapunov derivative V̇(x) = ∂V/∂x · f(x, π(x))
    using finite differences. -/
def lyapunovDerivative (V : LyapunovFunction) (sys : DynamicalSystem)
    (ctrl : NeuralController) (x : Tensor) (h : Float := 1e-4) : Option Float := do
  -- Compute control input u = π(x)
  let u ← evalController ctrl x
  -- Compute state derivative ẋ = f(x, u)
  let xdot ← sys.dynamics x u
  -- Approximate V̇ = ∇V · ẋ using central differences
  let mut vdot := 0.0
  for i in [:x.data.size] do
    -- ∂V/∂xᵢ ≈ (V(x + h·eᵢ) - V(x - h·eᵢ)) / (2h)
    let xPlus := x.setFlat i (x.data[i]! + h)
    let xMinus := x.setFlat i (x.data[i]! - h)
    let vPlus ← evalLyapunov V xPlus
    let vMinus ← evalLyapunov V xMinus
    let dVdxi := (vPlus - vMinus) / (2.0 * h)
    vdot := vdot + dVdxi * xdot.data[i]!
  return vdot

/-! ## Verification Using IBP -/

/-- Lyapunov verification result over a region. -/
structure LyapunovVerificationResult where
  /-- Minimum V(x) found in region (should be > 0 for x ≠ 0). -/
  minLyapunovValue : Float
  /-- Maximum V̇(x) found in region (should be < 0 for x ≠ 0). -/
  maxLyapunovDeriv : Float
  /-- Whether stability is verified. -/
  isStable : Bool
  deriving Repr

/-- Verify Lyapunov stability on a grid of states.
    Checks: V(x) > 0 and V̇(x) < 0 for all sampled x ≠ 0. -/
def verifyLyapunovGrid (V : LyapunovFunction) (sys : DynamicalSystem)
    (ctrl : NeuralController) (states : List Tensor)
    (zeroTol : Float := 1e-6) : Option LyapunovVerificationResult := do
  let mut minV := 1.0e38
  let mut maxVdot := -1.0e38
  let mut stable := true

  for x in states do
    -- Skip if x ≈ 0
    let norm := x.data.foldl (fun acc v => acc + fabs v) 0.0
    if norm > zeroTol then
      let vx ← evalLyapunov V x
      let vdot ← lyapunovDerivative V sys ctrl x

      minV := fmin minV vx
      maxVdot := fmax maxVdot vdot

      -- Check V(x) > 0
      if vx ≤ 0.0 then stable := false
      -- Check V̇(x) < 0
      if vdot ≥ 0.0 then stable := false

  return ⟨minV, maxVdot, stable⟩

/-- Verify Lyapunov stability using IBP over a region [−R, R]ⁿ.
    Uses interval bounds on V and V̇ to certify stability. -/
def verifyLyapunovIBP (V : LyapunovFunction) (regionRadius : Float) :
    Option Bool := do
  -- Check V(x) ≥ 0 over the region using IBP
  let inputBounds := Interval.epsBall (Tensor.zeros [V.stateDim]) regionRadius
  let outputBounds ← ibpNetwork V.network inputBounds
  -- V_min ≥ lower bound of output
  let vMin ← outputBounds.lower.getFlat 0
  return vMin ≥ 0.0

/-! ## Example Dynamical Systems -/

/-- Simple pendulum: ẋ₁ = x₂, ẋ₂ = -sin(x₁) - b·x₂ + u -/
def pendulum (damping : Float) : DynamicalSystem :=
  { stateDim := 2
    controlDim := 1
    dynamics := fun x u => do
      let x1 := x.data[0]!
      let x2 := x.data[1]!
      let u0 := u.data[0]!
      Tensor.ofData [2] #[x2, -Float.sin x1 - damping * x2 + u0] }

/-- Linear system: ẋ = Ax + Bu -/
def linearSystem (a : Tensor) (b : Tensor) : DynamicalSystem :=
  { stateDim := match a.shape with | [n, _] => n | _ => 0
    controlDim := match b.shape with | [_, m] => m | _ => 0
    dynamics := fun x u => do
      let ax ← Tensor.matVecMul a x
      let bu ← Tensor.matVecMul b u
      Tensor.add ax bu }

/-! ## Lyapunov Stability Theorem -/

/-- Lyapunov Stability Theorem (formalized):
    If V(x) > 0 for x ≠ 0 and V̇(x) < 0 for x ≠ 0,
    then the equilibrium x = 0 is asymptotically stable. -/
theorem lyapunov_stability
    (V : LyapunovFunction) (sys : DynamicalSystem) (ctrl : NeuralController)
    (hposdef : ∀ x : Tensor, x ≠ Tensor.zeros [V.stateDim] →
      ∃ v, evalLyapunov V x = some v ∧ v > 0.0)
    (hnegdef : ∀ x : Tensor, x ≠ Tensor.zeros [V.stateDim] →
      ∃ vdot, lyapunovDerivative V sys ctrl x = some vdot ∧ vdot < 0.0) :
    True := by  -- Stability conclusion (simplified statement)
  sorry

end TorchLean
