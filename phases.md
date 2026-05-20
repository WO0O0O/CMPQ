# CMPQ: Project Phases and Roadmap

## Phase 1: Foundation
* **Goal**: Environment Setup
* **Actions**: Install Lean 4, configure the `Mathlib` environment, and incorporate the **TorchLean** library.

## Phase 2: Modelling
* **Goal**: Semantics Definition
* **Actions**: Use TorchLean to define a single `Linear` layer and a custom `Quantise` operator with IEEE-754 semantics.

## Phase 3: The Proof
* **Goal**: Theorem Formulation
* **Actions**: Formalise the error bound: $\forall x, \| \text{Layer}(x) - \text{QuantisedLayer}(x) \| \leq \epsilon$.

## Phase 4: Verification
* **Goal**: Code Synthesis
* **Actions**: Use Lean 4 to write a "tactic" that automatically checks if a given quantisation policy satisfies the error bound theorem.
