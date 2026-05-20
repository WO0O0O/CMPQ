-- TorchLean.Runtime.Arith
-- Floating-point arithmetic utilities and activation functions

import TorchLean.Runtime.Float32

namespace TorchLean

/-! ## Float utility functions -/

/-- Absolute value. -/
def fabs (x : Float) : Float := if x < 0.0 then -x else x

/-- Maximum of two floats. -/
def fmax (a b : Float) : Float := if a ≥ b then a else b

/-- Minimum of two floats. -/
def fmin (a b : Float) : Float := if a ≤ b then a else b

/-- Clamp a value to [lo, hi]. -/
def fclamp (x lo hi : Float) : Float := fmax lo (fmin x hi)

/-! ## Activation functions -/

/-- ReLU activation: max(0, x) -/
def relu (x : Float) : Float := fmax 0.0 x

/-- Sigmoid activation: 1 / (1 + exp(-x))
    Numerically stable version. -/
def sigmoid (x : Float) : Float :=
  if x ≥ 0.0 then
    let ex := Float.exp (-x)
    1.0 / (1.0 + ex)
  else
    let ex := Float.exp x
    ex / (1.0 + ex)

/-- Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Implemented as 2·sigmoid(2x) - 1 for numerical stability. -/
def tanh' (x : Float) : Float :=
  2.0 * sigmoid (2.0 * x) - 1.0

/-! ## Derivative of activation functions (for CROWN) -/

/-- Derivative of ReLU. -/
def reluDeriv (x : Float) : Float := if x > 0.0 then 1.0 else 0.0

/-- Derivative of sigmoid. -/
def sigmoidDeriv (x : Float) : Float :=
  let s := sigmoid x
  s * (1.0 - s)

/-- Derivative of tanh. -/
def tanhDeriv (x : Float) : Float :=
  let t := tanh' x
  1.0 - t * t

/-! ## Vector operations on Float arrays -/

/-- Element-wise operation on two arrays. -/
def arrayZipWith (f : Float → Float → Float) (a b : Array Float) : Array Float := Id.run do
  let n := min a.size b.size
  let mut result := Array.mkEmpty n
  for i in [:n] do
    result := result.push (f a[i]! b[i]!)
  return result

/-- Element-wise map on an array. -/
def arrayMap (f : Float → Float) (a : Array Float) : Array Float :=
  a.map f

/-- Dot product of two Float arrays. -/
def dotProduct (a b : Array Float) : Float := Id.run do
  let n := min a.size b.size
  let mut sum := 0.0
  for i in [:n] do
    sum := sum + a[i]! * b[i]!
  return sum

/-- Sum of all elements. -/
def arraySum (a : Array Float) : Float :=
  a.foldl (· + ·) 0.0

/-- Matrix-vector multiply. mat is stored row-major as [rows][cols].
    Returns array of length `rows`. -/
def matVecMul (mat : Array (Array Float)) (vec : Array Float) : Array Float :=
  mat.map (dotProduct · vec)

/-- Vector addition. -/
def vecAdd (a b : Array Float) : Array Float :=
  arrayZipWith (· + ·) a b

/-- Vector subtraction. -/
def vecSub (a b : Array Float) : Array Float :=
  arrayZipWith (· - ·) a b

/-- Scalar-vector multiply. -/
def scalarVecMul (s : Float) (v : Array Float) : Array Float :=
  v.map (s * ·)

/-- Argmax: index of maximum element. -/
def argmax (a : Array Float) : Nat := Id.run do
  if a.size == 0 then return 0
  let mut maxIdx := 0
  let mut maxVal := a[0]!
  for i in [1:a.size] do
    if a[i]! > maxVal then
      maxIdx := i
      maxVal := a[i]!
  return maxIdx

end TorchLean
