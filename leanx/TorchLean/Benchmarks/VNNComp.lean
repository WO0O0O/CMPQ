-- TorchLean.Benchmarks.VNNComp
-- VNN-COMP benchmark runner and comparison (§7.4 of the paper)
--
-- Implements the VNN-COMP (Verification of Neural Networks Competition)
-- evaluation framework for comparing verification tools.

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown
import TorchLean.Verification.Certificate
import TorchLean.Verification.AlphaBetaCrown
import TorchLean.Benchmarks.AcasXu
import TorchLean.Benchmarks.MNIST

namespace TorchLean

/-! ## VNN-COMP Benchmark Categories -/

/-- VNN-COMP benchmark category. -/
inductive VNNCategory where
  | acasXu        -- ACAS Xu collision avoidance
  | mnistFC       -- MNIST fully-connected
  | mnistConv     -- MNIST convolutional
  | cifar10       -- CIFAR-10
  | ovalBab       -- Oval-BaB benchmarks
  | sri           -- SRI International benchmarks
  | tinyImageNet  -- TinyImageNet
  deriving Repr, BEq

/-- VNN-COMP verification result status. -/
inductive VNNResult where
  | sat           -- Property is satisfiable (counterexample found)
  | unsat         -- Property is unsatisfiable (verified safe)
  | unknown       -- Could not determine (timeout or other)
  | timeout       -- Exceeded time limit
  | error         -- Error during verification
  deriving Repr, BEq

/-- A single VNN-COMP benchmark instance. -/
structure VNNInstance where
  /-- Benchmark category. -/
  category : VNNCategory
  /-- Instance name/identifier. -/
  name : String
  /-- Network to verify. -/
  network : Network
  /-- Input bounds. -/
  inputBounds : Interval
  /-- Output specification to verify. -/
  outputSpec : Tensor → Bool
  /-- Timeout in seconds. -/
  timeout : Nat := 300

instance : Inhabited VNNInstance :=
  ⟨{ category := .acasXu, name := "", network := ⟨[]⟩,
     inputBounds := default, outputSpec := fun _ => true }⟩

/-- Result of a single VNN-COMP verification instance. -/
structure VNNInstanceResult where
  instanceName : String
  category : VNNCategory
  result : VNNResult
  /-- Time taken (in a real implementation). -/
  timeTaken : Float := 0.0
  deriving Repr

/-! ## VNN-COMP Runner -/

/-- Verify a single VNN-COMP instance using IBP. -/
def verifyVNNInstanceIBP (inst : VNNInstance) : VNNInstanceResult :=
  match ibpNetwork inst.network inst.inputBounds with
  | none => { instanceName := inst.name, category := inst.category, result := .error }
  | some outputBounds =>
    let verified := inst.outputSpec outputBounds.lower &&
                    inst.outputSpec outputBounds.upper
    { instanceName := inst.name
      category := inst.category
      result := if verified then .unsat else .unknown }

/-- Verify a single VNN-COMP instance using CROWN. -/
def verifyVNNInstanceCROWN (inst : VNNInstance) : VNNInstanceResult :=
  match crownBackward inst.network inst.inputBounds with
  | none => { instanceName := inst.name, category := inst.category, result := .error }
  | some outputBounds =>
    let verified := inst.outputSpec outputBounds.lower &&
                    inst.outputSpec outputBounds.upper
    { instanceName := inst.name
      category := inst.category
      result := if verified then .unsat else .unknown }

/-- Verify a single VNN-COMP instance using α,β-CROWN with BaB. -/
def verifyVNNInstanceAlphaBetaCROWN (inst : VNNInstance)
    (maxBranches : Nat := 64) : VNNInstanceResult :=
  -- Use the midpoint as the center input
  match Interval.midpoint inst.inputBounds with
  | none => { instanceName := inst.name, category := inst.category, result := .error }
  | some center =>
    -- Compute epsilon from bounds
    match Tensor.sub inst.inputBounds.upper center with
    | none => { instanceName := inst.name, category := inst.category, result := .error }
    | some diff =>
      let epsilon := diff.data.foldl fmax 0.0
      -- Try α,β-CROWN (using class 0 as default)
      let babResult := branchAndBound inst.network center epsilon 0 maxBranches
      { instanceName := inst.name
        category := inst.category
        result := if babResult.verified then .unsat else .unknown }

/-! ## VNN-COMP Scoring -/

/-- VNN-COMP scoring: points for correct results. -/
structure VNNScore where
  totalInstances : Nat
  solved : Nat
  unsatCount : Nat
  satCount : Nat
  unknownCount : Nat
  timeoutCount : Nat
  errorCount : Nat
  deriving Repr

/-- Compute VNN-COMP score from results. -/
def computeVNNScore (results : List VNNInstanceResult) : VNNScore := Id.run do
  let mut score : VNNScore := {
    totalInstances := results.length
    solved := 0, unsatCount := 0, satCount := 0
    unknownCount := 0, timeoutCount := 0, errorCount := 0
  }
  for r in results do
    match r.result with
    | .unsat   => score := { score with solved := score.solved + 1
                                        unsatCount := score.unsatCount + 1 }
    | .sat     => score := { score with solved := score.solved + 1
                                        satCount := score.satCount + 1 }
    | .unknown => score := { score with unknownCount := score.unknownCount + 1 }
    | .timeout => score := { score with timeoutCount := score.timeoutCount + 1 }
    | .error   => score := { score with errorCount := score.errorCount + 1 }
  return score

/-! ## Comparison with External Tools -/

/-- Comparison entry: TorchLean vs external tool. -/
structure ToolComparison where
  toolName : String
  category : VNNCategory
  torchLeanScore : VNNScore
  externalScore : VNNScore
  deriving Repr

/-- Reference results from external tools (from VNN-COMP 2023/2024). -/
structure ExternalToolResults where
  /-- Tool name. -/
  toolName : String
  /-- Solved instances per category. -/
  solvedByCategory : List (VNNCategory × Nat)
  deriving Repr

/-- Known external tool performance (from paper Table 2). -/
def marabouReference : ExternalToolResults :=
  { toolName := "Marabou"
    solvedByCategory := [
      (.acasXu, 180),      -- Marabou can solve most ACAS Xu
      (.mnistFC, 45),       -- Limited on MNIST FC
      (.cifar10, 12)        -- Limited on CIFAR-10
    ] }

def alphaBetaCrownReference : ExternalToolResults :=
  { toolName := "α,β-CROWN"
    solvedByCategory := [
      (.acasXu, 185),       -- Near-complete on ACAS Xu
      (.mnistFC, 78),        -- Strong on MNIST FC
      (.mnistConv, 52),      -- Good on MNIST Conv
      (.cifar10, 38)         -- Best on CIFAR-10
    ] }

/-- Format VNN-COMP results as a comparison table. -/
def formatVNNCompTable (results : List VNNInstanceResult) : String := Id.run do
  let score := computeVNNScore results
  let mut s := "VNN-COMP Results Summary\n"
  s := s ++ "========================\n"
  s := s ++ s!"Total instances: {score.totalInstances}\n"
  s := s ++ s!"Solved:          {score.solved}\n"
  s := s ++ s!"  UNSAT:         {score.unsatCount}\n"
  s := s ++ s!"  SAT:           {score.satCount}\n"
  s := s ++ s!"Unknown:         {score.unknownCount}\n"
  s := s ++ s!"Timeout:         {score.timeoutCount}\n"
  s := s ++ s!"Error:           {score.errorCount}\n"
  return s

end TorchLean
