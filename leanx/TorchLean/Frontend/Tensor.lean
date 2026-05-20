-- TorchLean.Frontend.Tensor
-- Practical tensor type with Float elements

import TorchLean.Runtime.Arith

namespace TorchLean

/-- Compute the total number of elements from a shape. -/
def shapeSize (shape : List Nat) : Nat :=
  shape.foldl (· * ·) 1

/-- A tensor with a given shape and Float data stored in row-major order. -/
structure Tensor where
  shape : List Nat
  data : Array Float
  deriving Repr, Inhabited

namespace Tensor

/-! ## Creation -/

/-- Create a tensor filled with a constant value. -/
def fill (shape : List Nat) (val : Float) : Tensor :=
  ⟨shape, Array.replicate (shapeSize shape) val⟩

/-- Create a tensor of zeros. -/
def zeros (shape : List Nat) : Tensor := fill shape 0.0

/-- Create a tensor of ones. -/
def ones (shape : List Nat) : Tensor := fill shape 1.0

/-- Create a scalar tensor. -/
def scalar (val : Float) : Tensor := ⟨[], #[val]⟩

/-- Create a 1-D tensor from a list. -/
def ofList (vals : List Float) : Tensor :=
  ⟨[vals.length], vals.toArray⟩

/-- Create a tensor from raw shape and data (validates size). -/
def ofData (shape : List Nat) (data : Array Float) : Option Tensor :=
  if data.size == shapeSize shape then
    some ⟨shape, data⟩
  else
    none

/-- Number of dimensions. -/
def ndim (t : Tensor) : Nat := t.shape.length

/-- Total number of elements. -/
def numel (t : Tensor) : Nat := t.data.size

/-! ## Element access -/

/-- Get element by flat index. -/
def getFlat (t : Tensor) (i : Nat) : Option Float :=
  if i < t.data.size then some t.data[i]! else none

/-- Set element by flat index. -/
def setFlat (t : Tensor) (i : Nat) (val : Float) : Tensor :=
  if i < t.data.size then
    ⟨t.shape, t.data.set! i val⟩
  else
    t

/-! ## Element-wise operations -/

/-- Apply a function element-wise to a tensor. -/
def map (f : Float → Float) (t : Tensor) : Tensor :=
  ⟨t.shape, t.data.map f⟩

/-- Apply a function element-wise to two tensors of the same shape. -/
def zipWith (f : Float → Float → Float) (a b : Tensor) : Option Tensor :=
  if a.shape == b.shape then
    some ⟨a.shape, arrayZipWith f a.data b.data⟩
  else
    none

/-- Element-wise addition. -/
def add (a b : Tensor) : Option Tensor := zipWith (· + ·) a b

/-- Element-wise subtraction. -/
def sub (a b : Tensor) : Option Tensor := zipWith (· - ·) a b

/-- Element-wise multiplication (Hadamard product). -/
def mul (a b : Tensor) : Option Tensor := zipWith (· * ·) a b

/-- Negate all elements. -/
def neg (t : Tensor) : Tensor := map (- ·) t

/-- Scalar multiplication. -/
def scalarMul (s : Float) (t : Tensor) : Tensor := map (s * ·) t

/-- Element-wise maximum. -/
def maxWith (a b : Tensor) : Option Tensor := zipWith fmax a b

/-- Element-wise minimum. -/
def minWith (a b : Tensor) : Option Tensor := zipWith fmin a b

/-- Element-wise max with zero (ReLU). -/
def reluT (t : Tensor) : Tensor := map relu t

/-! ## 2-D Matrix operations -/

/-- Matrix-vector multiply: [m, n] × [n] → [m] -/
def matVecMul (mat : Tensor) (vec : Tensor) : Option Tensor :=
  match mat.shape, vec.shape with
  | [m, n], [n'] =>
    if n == n' then Id.run do
      let mut result := Array.mkEmpty m
      for i in [:m] do
        let mut sum := 0.0
        for j in [:n] do
          sum := sum + mat.data[i * n + j]! * vec.data[j]!
        result := result.push sum
      return some ⟨[m], result⟩
    else none
  | _, _ => none

/-- Matrix-matrix multiply: [m, k] × [k, n] → [m, n] -/
def matMul (a b : Tensor) : Option Tensor :=
  match a.shape, b.shape with
  | [m, k], [k', n] =>
    if k == k' then Id.run do
      let mut result := Array.mkEmpty (m * n)
      for i in [:m] do
        for j in [:n] do
          let mut sum := 0.0
          for l in [:k] do
            sum := sum + a.data[i * k + l]! * b.data[l * n + j]!
          result := result.push sum
      return some ⟨[m, n], result⟩
    else none
  | _, _ => none

/-- Transpose a 2-D tensor [m, n] → [n, m] -/
def transpose (t : Tensor) : Option Tensor :=
  match t.shape with
  | [m, n] => Id.run do
    let mut result := Array.mkEmpty (m * n)
    for j in [:n] do
      for i in [:m] do
        result := result.push t.data[i * n + j]!
    return some ⟨[n, m], result⟩
  | _ => none

/-! ## Reduction operations -/

/-- Sum of all elements. -/
def sum (t : Tensor) : Float := arraySum t.data

/-- Maximum element. -/
def maxElem (t : Tensor) : Option Float :=
  if t.data.size == 0 then none
  else some (t.data.foldl fmax t.data[0]!)

/-- Minimum element. -/
def minElem (t : Tensor) : Option Float :=
  if t.data.size == 0 then none
  else some (t.data.foldl fmin t.data[0]!)

/-- Argmax: index of maximum element. -/
def argmaxT (t : Tensor) : Nat := TorchLean.argmax t.data

/-! ## Display -/

/-- Convert tensor to string for display. -/
def toString (t : Tensor) : String :=
  let dataStr := t.data.toList.map Float.toString |>.intersperse ", " |> String.join
  s!"Tensor(shape={t.shape}, data=[{dataStr}])"

instance : ToString Tensor := ⟨Tensor.toString⟩

end Tensor
end TorchLean
