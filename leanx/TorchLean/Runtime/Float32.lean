-- TorchLean.Runtime.Float32
-- IEEE-754 binary32 formalization and executable semantics

namespace TorchLean

/-- IEEE-754 binary32 floating-point representation. -/
structure Float32 where
  sign : Bool
  exponent : Fin 256       -- 8-bit exponent (0..255)
  mantissa : Fin 8388608   -- 23-bit mantissa (0..2^23-1)
  deriving Repr, BEq

namespace Float32

/-- Exponent bias for binary32. -/
def bias : Nat := 127

/-- Machine epsilon for binary32: 2^(-23) ≈ 1.19e-7 -/
def machineEpsilon : Float := 1.1920929e-7

/-- Positive zero. -/
def posZero : Float32 := ⟨false, ⟨0, by omega⟩, ⟨0, by omega⟩⟩

/-- Negative zero. -/
def negZero : Float32 := ⟨true, ⟨0, by omega⟩, ⟨0, by omega⟩⟩

/-- Positive infinity. -/
def posInf : Float32 := ⟨false, ⟨255, by omega⟩, ⟨0, by omega⟩⟩

/-- Negative infinity. -/
def negInf : Float32 := ⟨true, ⟨255, by omega⟩, ⟨0, by omega⟩⟩

/-- Check if the value is NaN (exponent=255, mantissa≠0). -/
def isNaN (f : Float32) : Bool :=
  f.exponent.val == 255 && f.mantissa.val != 0

/-- Check if the value is infinity (exponent=255, mantissa=0). -/
def isInf (f : Float32) : Bool :=
  f.exponent.val == 255 && f.mantissa.val == 0

/-- Check if the value is subnormal (exponent=0, mantissa≠0). -/
def isSubnormal (f : Float32) : Bool :=
  f.exponent.val == 0 && f.mantissa.val != 0

/-- Check if the value is zero (exponent=0, mantissa=0). -/
def isZero (f : Float32) : Bool :=
  f.exponent.val == 0 && f.mantissa.val == 0

/-- Check if the value is a normal number. -/
def isNormal (f : Float32) : Bool :=
  f.exponent.val != 0 && f.exponent.val != 255

/-- Convert Float32 to Lean's Float (for computation).
    Normal:    (-1)^s × 2^(e-127) × (1 + m/2^23)
    Subnormal: (-1)^s × 2^(-126) × (m/2^23) -/
def toFloat (f : Float32) : Float :=
  if f.isNaN then 0.0 / 0.0  -- NaN
  else if f.isInf then
    if f.sign then -1.0 / 0.0 else 1.0 / 0.0
  else if f.isZero then 0.0
  else
    let s : Float := if f.sign then -1.0 else 1.0
    let m : Float := Float.ofNat f.mantissa.val / 8388608.0  -- / 2^23
    if f.isSubnormal then
      -- (-1)^s × 2^(-126) × (m/2^23)
      s * (2.0 ^ (-126.0 : Float)) * m
    else
      -- (-1)^s × 2^(e-127) × (1 + m/2^23)
      let e : Float := Float.ofNat f.exponent.val - 127.0
      s * (2.0 ^ e) * (1.0 + m)

/-- Category of a Float32 value. -/
inductive Category where
  | zero
  | subnormal
  | normal
  | infinity
  | nan
  deriving Repr, BEq

/-- Get the category of a Float32 value. -/
def category (f : Float32) : Category :=
  if f.isNaN then .nan
  else if f.isInf then .infinity
  else if f.isZero then .zero
  else if f.isSubnormal then .subnormal
  else .normal

end Float32

/-- IEEE-754 rounding modes. -/
inductive RoundingMode where
  | roundNearestEven    -- Default: round to nearest, ties to even
  | roundTowardPositive -- Round toward +∞
  | roundTowardNegative -- Round toward -∞
  | roundTowardZero     -- Round toward zero
  deriving Repr, BEq

end TorchLean
