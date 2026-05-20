-- TorchLean.Benchmarks.MNIST
-- MNIST and CIFAR-10 benchmark specifications (§7.2, §7.3 of the paper)
--
-- Standard image classification benchmarks for neural network verification.

import TorchLean.Frontend.Tensor
import TorchLean.Frontend.Layers
import TorchLean.Verification.IBP
import TorchLean.Verification.Crown
import TorchLean.Verification.Certificate

namespace TorchLean

/-! ## MNIST Benchmark Specification -/

/-- MNIST dataset parameters. -/
structure MNISTSpec where
  /-- Image height. -/
  imageH : Nat := 28
  /-- Image width. -/
  imageW : Nat := 28
  /-- Number of channels (grayscale). -/
  channels : Nat := 1
  /-- Number of classes. -/
  numClasses : Nat := 10
  /-- Input dimension (flattened). -/
  inputDim : Nat := 784
  /-- Pixel value range. -/
  pixelMin : Float := 0.0
  pixelMax : Float := 1.0
  deriving Repr, Inhabited

/-- Standard epsilon values for MNIST verification. -/
def mnistEpsilons : List Float := [0.01, 0.02, 0.03, 0.05, 0.1, 0.3]

/-- Build a small MNIST classifier for verification.
    Architecture: 784 → 256 → ReLU → 128 → ReLU → 10 -/
def buildMNISTSmall : Network := Id.run do
  let mut net := Network.empty
  net := net.addLinearAct (LinearLayer.zeros 784 256) .relu
  net := net.addLinearAct (LinearLayer.zeros 256 128) .relu
  net := net.addLayer (.linear (LinearLayer.zeros 128 10))
  return net

/-- Build a medium MNIST classifier.
    Architecture: 784 → 512 → ReLU → 256 → ReLU → 128 → ReLU → 10 -/
def buildMNISTMedium : Network := Id.run do
  let mut net := Network.empty
  net := net.addLinearAct (LinearLayer.zeros 784 512) .relu
  net := net.addLinearAct (LinearLayer.zeros 512 256) .relu
  net := net.addLinearAct (LinearLayer.zeros 256 128) .relu
  net := net.addLayer (.linear (LinearLayer.zeros 128 10))
  return net

/-- Build a convolutional MNIST classifier.
    Architecture: Conv(1,16,5) → ReLU → MaxPool(2) → Conv(16,32,5) → ReLU → MaxPool(2) →
                  Flatten → 512 → ReLU → 10 -/
def buildMNISTConv : Network := Id.run do
  let mut net := Network.empty
  -- Conv1: 1 channel → 16 channels, 5×5 kernel
  let conv1Kernel := Tensor.zeros [16, 1, 5, 5]
  let conv1Bias := Tensor.zeros [16]
  net := net.addLayer (.conv2d ⟨1, 16, 5, 5, 1, 1, 0, 0, conv1Kernel, conv1Bias⟩)
  net := net.addLayer (.activation .relu)
  net := net.addLayer (.maxPool2d ⟨2, 2, 2, 2, 0, 0⟩)
  -- Conv2: 16 channels → 32 channels, 5×5 kernel
  let conv2Kernel := Tensor.zeros [32, 16, 5, 5]
  let conv2Bias := Tensor.zeros [32]
  net := net.addLayer (.conv2d ⟨16, 32, 5, 5, 1, 1, 0, 0, conv2Kernel, conv2Bias⟩)
  net := net.addLayer (.activation .relu)
  net := net.addLayer (.maxPool2d ⟨2, 2, 2, 2, 0, 0⟩)
  -- Flatten and FC layers
  net := net.addLayer .flatten
  net := net.addLinearAct (LinearLayer.zeros 512 256) .relu
  net := net.addLayer (.linear (LinearLayer.zeros 256 10))
  return net

/-! ## CIFAR-10 Benchmark Specification -/

/-- CIFAR-10 dataset parameters. -/
structure CIFAR10Spec where
  imageH : Nat := 32
  imageW : Nat := 32
  channels : Nat := 3
  numClasses : Nat := 10
  inputDim : Nat := 3072  -- 3 × 32 × 32
  pixelMin : Float := 0.0
  pixelMax : Float := 1.0
  deriving Repr, Inhabited

/-- Standard epsilon values for CIFAR-10 verification (typically smaller). -/
def cifar10Epsilons : List Float := [1.0/255.0, 2.0/255.0, 4.0/255.0, 8.0/255.0]

/-- Build a small CIFAR-10 classifier.
    Architecture: 3072 → 512 → ReLU → 256 → ReLU → 10 -/
def buildCIFAR10Small : Network := Id.run do
  let mut net := Network.empty
  net := net.addLinearAct (LinearLayer.zeros 3072 512) .relu
  net := net.addLinearAct (LinearLayer.zeros 512 256) .relu
  net := net.addLayer (.linear (LinearLayer.zeros 256 10))
  return net

/-- Build a convolutional CIFAR-10 classifier.
    Architecture: Conv(3,32,3,pad=1) → ReLU → Conv(32,32,3,pad=1) → ReLU →
                  MaxPool(2) → Conv(32,64,3,pad=1) → ReLU →
                  Conv(64,64,3,pad=1) → ReLU → MaxPool(2) →
                  Flatten → 4096 → ReLU → 512 → ReLU → 10 -/
def buildCIFAR10Conv : Network := Id.run do
  let mut net := Network.empty
  -- Block 1
  let k1 := Tensor.zeros [32, 3, 3, 3]
  let b1 := Tensor.zeros [32]
  net := net.addLayer (.conv2d ⟨3, 32, 3, 3, 1, 1, 1, 1, k1, b1⟩)
  net := net.addLayer (.activation .relu)
  let k2 := Tensor.zeros [32, 32, 3, 3]
  let b2 := Tensor.zeros [32]
  net := net.addLayer (.conv2d ⟨32, 32, 3, 3, 1, 1, 1, 1, k2, b2⟩)
  net := net.addLayer (.activation .relu)
  net := net.addLayer (.maxPool2d ⟨2, 2, 2, 2, 0, 0⟩)
  -- Block 2
  let k3 := Tensor.zeros [64, 32, 3, 3]
  let b3 := Tensor.zeros [64]
  net := net.addLayer (.conv2d ⟨32, 64, 3, 3, 1, 1, 1, 1, k3, b3⟩)
  net := net.addLayer (.activation .relu)
  let k4 := Tensor.zeros [64, 64, 3, 3]
  let b4 := Tensor.zeros [64]
  net := net.addLayer (.conv2d ⟨64, 64, 3, 3, 1, 1, 1, 1, k4, b4⟩)
  net := net.addLayer (.activation .relu)
  net := net.addLayer (.maxPool2d ⟨2, 2, 2, 2, 0, 0⟩)
  -- FC layers
  net := net.addLayer .flatten
  net := net.addLinearAct (LinearLayer.zeros 4096 512) .relu
  net := net.addLayer (.linear (LinearLayer.zeros 512 10))
  return net

/-! ## Benchmark Runner -/

/-- Benchmark result for a single model + epsilon combination. -/
structure BenchmarkResult where
  modelName : String
  epsilon : Float
  numSamples : Nat
  verifiedCount : Nat
  verifiedRate : Float
  method : String
  deriving Repr

/-- Run verification benchmark on a set of test inputs. -/
def runVerificationBenchmark (net : Network) (modelName : String)
    (inputs : List Tensor) (labels : List Nat) (epsilon : Float)
    (method : VerificationMethod := .ibp) : Option BenchmarkResult := do
  let rate ← batchVerify net inputs epsilon labels method
  let verifiedCount := (rate * Float.ofNat inputs.length).toUInt64.toNat
  return {
    modelName := modelName
    epsilon := epsilon
    numSamples := inputs.length
    verifiedCount := verifiedCount
    verifiedRate := rate
    method := match method with | .ibp => "IBP" | .crown => "CROWN" | .alphaBetaCrown => "α,β-CROWN"
  }

/-- Generate a synthetic test sample for MNIST (zeros image). -/
def syntheticMNISTSample (label : Nat) : Tensor × Nat :=
  (Tensor.zeros [784], label)

/-- Generate synthetic test batch. -/
def syntheticMNISTBatch (n : Nat) : List Tensor × List Nat := Id.run do
  let mut inputs : List Tensor := []
  let mut labels : List Nat := []
  for i in [:n] do
    let (inp, lbl) := syntheticMNISTSample (i % 10)
    inputs := inputs ++ [inp]
    labels := labels ++ [lbl]
  return (inputs, labels)

/-! ## Benchmark Comparison Table -/

/-- Format benchmark results as a table. -/
def formatBenchmarkTable (results : List BenchmarkResult) : String := Id.run do
  let mut s := "Model           | ε        | Method | Verified | Rate\n"
  s := s ++ "----------------|----------|--------|----------|------\n"
  for r in results do
    let padLen := if r.modelName.length < 16 then 16 - r.modelName.length else 0
    let modelPad := r.modelName ++ String.ofList (List.replicate padLen ' ')
    s := s ++ s!"{modelPad}| {r.epsilon} | {r.method}   | {r.verifiedCount}/{r.numSamples}      | {r.verifiedRate}\n"
  return s

end TorchLean
