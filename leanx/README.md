# TorchLean

**Formalizing Neural Networks in Lean 4** &mdash; A mechanized framework for defining, executing, and formally verifying neural networks.

Based on the paper: [*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631) (R. J. George, J. Cruden, X. Zhong, H. Zhang, A. Anandkumar, 2026).

## Overview

TorchLean bridges the gap between deep learning and formal verification by providing:

- **PyTorch-style API** for defining neural networks in Lean 4
- **IEEE-754 binary32** floating-point formalization with three-level semantics
- **Interval Bound Propagation (IBP)** for all layer types (Linear, Conv2d, BatchNorm, MaxPool, AvgPool)
- **CROWN/LiRPA** linear relaxation bound propagation
- **&alpha;,&beta;-CROWN** with branch-and-bound for complete verification
- **Adversarial attacks** (FGSM, PGD) for empirical robustness testing
- **Formal robustness theorems** and verification certificates
- **Application case studies**: PINN verification, Lyapunov stability, Universal Approximation Theorem

## Architecture

```
TorchLean/
├── Runtime/            § 4 — IEEE-754 Floating-Point Semantics
│   ├── Float32.lean         IEEE-754 binary32 representation
│   ├── Arith.lean           Activation functions & vector operations
│   └── Semantics.lean       Three-level semantics (Abstract/Concrete/Verified)
│
├── Frontend/           § 3 — PyTorch-Style Neural Network API
│   ├── Tensor.lean          Tensor type with element-wise & matrix operations
│   ├── Layers.lean          Linear, Conv2d, BatchNorm, MaxPool2d, AvgPool2d, Dropout
│   ├── Graph.lean           Op-tagged SSA/DAG computation graph IR
│   └── Execution.lean       Eager/Compiled modes, ONNX-like import
│
├── Verification/       § 5 — Bound Propagation & Verification
│   ├── IBP.lean             Interval Bound Propagation (W⁺/W⁻ decomposition)
│   ├── Crown.lean           CROWN/LiRPA backward linear bounds
│   ├── AlphaBetaCrown.lean  α,β-CROWN with branch-and-bound
│   ├── Robustness.lean      Formal robustness definitions & theorems
│   ├── Certificate.lean     Verification certificate generation & checking
│   ├── Attacks.lean         FGSM & PGD adversarial attacks
│   └── Tactics.lean         ReLU stability analysis & adaptive verification
│
├── Applications/       § 6 — Case Studies
│   ├── PINN.lean            Physics-Informed Neural Network verification (Burgers' eq.)
│   ├── Lyapunov.lean        Neural controller Lyapunov stability
│   └── UniversalApprox.lean Universal Approximation Theorem for ReLU networks
│
└── Benchmarks/         § 7 — Benchmark Infrastructure
    ├── AcasXu.lean          ACAS Xu safety properties (φ₁–φ₅)
    ├── MNIST.lean           MNIST & CIFAR-10 model specifications
    └── VNNComp.lean         VNN-COMP evaluation framework
```

## Requirements

- [Lean 4](https://leanprover.github.io/lean4/doc/quickstart.html) (v4.28.0)
- [Lake](https://github.com/leanprover/lake) (bundled with Lean)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/torchlean.git
cd torchlean

# Build the project
lake build

# Run the end-to-end demo
lake exe torchlean
```

## Demo Output

The demo showcases all major features:

```
╔══════════════════════════════════════════╗
║  TorchLean: Neural Network Verification  ║
╚══════════════════════════════════════════╝

── 1. Network Construction ──
  Layers: 3, Parameters: 22

── 2. Forward Pass ──
  Input:  Tensor(shape=[2], data=[0.5, 0.8])
  Output: Tensor(shape=[2], data=[0.448, 0.049])

── 4. IBP Robustness Verification ──
  ε = 0.05, IBP robust: true

── 5. CROWN Robustness Verification ──
  CROWN robust: true (tighter bounds than IBP)

── 6. α,β-CROWN Verification ──
  Verified: true, Branches: 1

── 8. Method Comparison ──
  IBP total output width:   0.284
  CROWN total output width: 0.200
  → CROWN provides tighter bounds

── 9. Adversarial Attacks ──
  FGSM: successful=false  (attack failed → robust)
  PGD:  successful=false  (attack failed → robust)
```

## Key Features

### Verification Methods

| Method | Type | Tightness | Speed |
|--------|------|-----------|-------|
| IBP | Incomplete | Loosest | Fastest |
| CROWN | Incomplete | Tighter | Medium |
| &alpha;,&beta;-CROWN + BaB | Complete | Tightest | Slowest |

### Supported Layer Types

| Layer | Forward | IBP | CROWN |
|-------|---------|-----|-------|
| Linear (FC) | ✓ | ✓ | ✓ |
| Conv2d | ✓ | ✓ | — |
| BatchNorm | ✓ | ✓ | — |
| MaxPool2d | ✓ | ✓ | — |
| AvgPool2d | ✓ | ✓ | — |
| ReLU | ✓ | ✓ | ✓ |
| Sigmoid | ✓ | ✓ | — |
| Tanh | ✓ | ✓ | — |
| Dropout | ✓ (id) | ✓ | ✓ |
| Flatten | ✓ | ✓ | ✓ |

### Formal Theorems

The following theorems are stated with their formal specifications:

- **IBP Soundness** (Theorem 1): IBP output bounds contain all reachable outputs
- **CROWN Soundness**: CROWN linear relaxation bounds are sound
- **CROWN ≥ IBP**: CROWN provides bounds at least as tight as IBP
- **Robustness via IBP/CROWN**: Verified bounds imply ε-robustness
- **&alpha;-CROWN ≥ CROWN**: Optimizable relaxation is at least as tight
- **BaB Completeness**: Branch-and-bound is a complete verification method
- **Lyapunov Stability**: V(x)>0 ∧ V̇(x)<0 ⟹ asymptotic stability
- **Universal Approximation**: ReLU networks can approximate any continuous function
- **Yarotsky Bound**: Depth O(log(1/ε)), width O(n·(1/ε)^(n/2))
- **PINN Verification**: Physics + boundary conditions ⟹ PDE residual bound
- **Semantic Refinement Chain**: Abstract ⊇ Concrete ⊇ Verified (**proved**, no `sorry`)

> **Note**: Most theorems involving `Float` arithmetic use `sorry` as Lean 4's native `Float` type is opaque. Full proofs require [Mathlib](https://github.com/leanprover-community/mathlib4) integration with real-number analysis.

## Project Statistics

| Metric | Value |
|--------|-------|
| Lean files | 28 |
| Total lines | 4,070 |
| Build jobs | 58 |
| Layer types | 10 |
| Verification methods | 3 (IBP, CROWN, α,β-CROWN) |
| Formal theorems | 20 |
| Benchmarks | 3 (ACAS Xu, MNIST/CIFAR-10, VNN-COMP) |

## Paper Sections Coverage

| Section | Topic | Status |
|---------|-------|--------|
| §3 | Frontend (PyTorch-style API) | ✓ Implemented |
| §4 | Runtime (IEEE-754 semantics) | ✓ Implemented |
| §5 | Verification (IBP, CROWN, α,β-CROWN) | ✓ Implemented |
| §5.3 | Verification Certificates | ✓ Implemented |
| §5.4 | Adversarial Attacks (FGSM, PGD) | ✓ Implemented |
| §6.1 | Robustness Verification | ✓ Implemented |
| §6.2 | PINN Verification | ✓ Implemented |
| §6.3 | Lyapunov Stability | ✓ Implemented |
| §6.4 | Universal Approximation | ✓ Implemented |
| §7 | Benchmarks (ACAS Xu, MNIST, VNN-COMP) | ✓ Implemented |

## Citation

```bibtex
@article{george2026torchlean,
  title={TorchLean: Formalizing Neural Networks in Lean},
  author={George, R. J. and Cruden, J. and Zhong, X. and Zhang, H. and Anandkumar, A.},
  journal={arXiv preprint arXiv:2602.22631},
  year={2026}
}
```

## License

MIT
