# qTransformer — Quantum Attention for Many-Body Physics

Can quantum circuits compute better attention than classical transformers? This project benchmarks **quantum vs. classical attention mechanisms** as neural network ansätze for approximating quantum ground states.

## The Problem (30-second version)

We're solving a quantum physics optimization problem: **find the lowest-energy state of a system of interacting spins on a lattice.** Think of it as a very hard combinatorial optimization over exponentially large state spaces (2^N configurations for N spins).

The approach: train a neural network to output `log ψ(x)` — the log-amplitude of the quantum wave function — for each spin configuration `x ∈ {+1, -1}^N`. This is called a **Neural Quantum State (NQS)**. Training uses **Variational Monte Carlo (VMC)**: sample spin configurations, evaluate the energy, and minimize it via gradient descent.

The question: does replacing the classical attention layer in a Vision Transformer with a **quantum circuit** (PQC) improve results?

## Models We Compare

```
Level 0  Simplified ViT     Position-only attention (no Q/K)
Level 1  Classical ViT       Standard dot-product Q·K attention
Level 2  QSANN               Quantum-encoded features, classical similarity
Level 3  QMSAN               Fully quantum similarity (swap test in Hilbert space)

+ baselines: RBM (simple), CNN+ResNet (convolutional), ED/DMRG (exact references)
```

## Quick Start

```bash
# Install
git clone <repo> && cd qTransformer_on_frustrated_Heisenberg_model
pip install -e .

# Run a test experiment (3 ED + 21 VMC tasks on 10-site chain)
python main.py test

# See what it does without running
python main.py test --dry-run

# On SLURM cluster
sbatch scripts/run_experiment.sbatch test
```

## How It Works

```
python main.py test
  │
  ├─ Reads conf/experiment/test.yaml          ← list of tasks
  │    tasks:
  │      - solution: rbm, hamiltonian: chain_10, g: 0.0
  │      - solution: qmsan_pure, hamiltonian: chain_10, g: 0.5
  │      ...
  │
  ├─ For each task, Hydra composes a config from:
  │    conf/solution/rbm.yaml        ← model architecture
  │    conf/hamiltonian/chain_10.yaml ← physics system
  │    conf/training/medium.yaml      ← optimizer settings
  │
  └─ Runs VMC training → saves results to outputs/
```

## Project Structure

```
main.py                      Entry point (positional arg: experiment name)
conf/
  experiment/                Task lists (test.yaml, baseline.yaml, benchmark.yaml)
  solution/                  Model configs (rbm.yaml, qmsan_pure.yaml, ...)
  hamiltonian/               Physics systems (chain_10.yaml, square_4x4.yaml)
  training/                  Optimizer settings (fast.yaml, medium.yaml, production.yaml)
src/
  experiment/
    experiment.py            Orchestrator — reads task list, resolves configs via Hydra
    task.py                  Task runner — dispatches to ED, DMRG, or VMC
  models/
    base_model.py            Interface: x ∈ {±1}^N → log ψ(x) ∈ ℂ
    factory.py               Model registry
    classical_models/        RBM, CNN-ResNet, Simplified ViT, Classical ViT
    quantum_models/          QSANN, QMSAN (the research contribution)
    training/
      vmc_runner.py          VMC training loop (NetKet)
      sr_optimizer.py        Stochastic Reconfiguration (natural gradient)
      callbacks.py           Logging, checkpointing, early stopping
  hamiltonians/              J1-J2 Heisenberg model (chain + square lattice)
  numerical_solvers/         ED (exact), DMRG (tensor network reference)
  evaluation/                Energy, entanglement, parameter efficiency metrics
scripts/                     SLURM scripts, utility commands
outputs/                     Results per experiment (gitignored)
text_docs/                   Proposal, detailed introduction, cluster notes
```

## Key Concepts for AI Collaborators

**VMC (Variational Monte Carlo):** Sample spin configs via MCMC → compute energy → update neural network params to minimize energy. Like training a generative model where the loss is the physical energy.

**SR (Stochastic Reconfiguration):** The optimizer. It's natural gradient descent — uses the Fisher information matrix to precondition gradients. Cost is O(P²) where P = number of params, so **fewer parameters = much cheaper training**.

**Frustration ratio g:** Controls problem difficulty. `g=0` is easy (ordered ground state), `g=0.5` is hard (competing interactions, no one knows the exact answer for large systems).

**Holomorphic:** Whether the model has complex-valued parameters. RBM does (holomorphic=True), quantum circuits don't (holomorphic=False). This affects how gradients are computed. Auto-detected by NetKet.

## Config System

All hyperparameters live in YAML files under `conf/`. To tune a model, edit its solution YAML:

```yaml
# conf/solution/classical_vit.yaml
name: classical_vit
type: classical_vit
d_model: 64      # embedding dimension
n_heads: 4       # attention heads
n_layers: 2      # transformer layers
d_ff: 128        # feedforward hidden dim
```

Training settings are separate:

```yaml
# conf/training/medium.yaml
n_steps: 200
learning_rate: 0.01
n_samples: 512
sr:
  diag_shift: 0.01   # SR regularization (larger = more stable, slower)
```

## Tech Stack

| Tool | Role |
|------|------|
| JAX + Flax | Neural network framework (autodiff, JIT) |
| NetKet | VMC engine — MCMC sampling + SR optimizer |
| PennyLane | Quantum circuit simulation (QSANN/QMSAN) |
| Hydra + OmegaConf | Config composition and validation |
| MLflow | Experiment tracking |
| QuSpin / TeNPy | Reference solvers (ED / DMRG) |

## References

- [Proposal](text_docs/proposal.md) — Full research proposal with complexity analysis
- [Project Introduction](text_docs/project_introduction.md) — Detailed architecture walkthrough
- [Experiments Plan](text_docs/experiments_plan.md) — Experimental design and phases
