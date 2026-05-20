# Certified Mixed-Precision Quantisation (CMPQ)

This is my project on formally verifying neural network quantisation. Right now, most model quantisation relies on empirical testing and heuristics to make sure accuracy doesn't drop too much. I want to replace that with actual mathematical guarantees. 

This project builds on the [TorchLean](https://github.com/nktkt/leanx) framework and the paper *TorchLean: Formalizing Neural Networks in Lean* (George et al., 2026). Using Lean 4, I'm aiming to compute a quantisation policy for a layer and generate a mathematical certificate to prove the output error is bounded by a specific $\epsilon$.


## End Goal
A verifiable Lean 4 tool that takes a neural network layer, applies my quantisation definitions, and outputs a formal proof certificate confirming the quantisation is mathematically safe.

## why am I doing this
I dont acc know

## hello there
General Kenobi
