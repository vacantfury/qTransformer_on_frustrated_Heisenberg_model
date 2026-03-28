# Quantum Transformer NQS for the Frustrated J1-J2 Heisenberg Model

A modular research codebase for studying hybrid quantum-classical attention mechanisms as neural quantum state (NQS) ansätze for the 2D frustrated Heisenberg model. This project implements a **spectrum of attention mechanisms** — from position-only to fully quantum swap-test — and benchmarks them within a Variational Monte Carlo (VMC) framework.

## Research Question

> Does quantum-native attention capture frustrated many-body correlations more parameter-efficiently than classical attention?

We compare 8 solutions organised into four tiers (see [proposal.md §4](proposal.md) for full details):

| Tier | Solution | Full Name | Method |
|------|----------|-----------|--------|
| 0 — Exact | **ED** | Exact Diagonalisation | Lanczos on full 2^N Hilbert space |
| 0 — Exact | **DMRG** | Density Matrix Renormalisation Group | Matrix Product State approximation |
| 1 — Classical NQS | **RBM** | Restricted Boltzmann Machine | Fully-connected single hidden layer |
| 1 — Classical NQS | **CNN+ResNet** | Convolutional NN + Residual Blocks | Local spatial correlations |
| 2 — Classical Attn | **Simplified ViT** | Simplified Vision Transformer | Position-only attention (Level 0) |
| 2 — Classical Attn | **Classical ViT** | Classical Vision Transformer | Dot-product Q·K attention (Level 1) |
| 3 — **Quantum Attn** | **QSANN** | Quantum Self-Attention NN | Gaussian-projected PQC attention (Level 2) |
| 3 — **Quantum Attn** | **QMSAN** | Quantum Mixed-State Self-Attention | Swap-test quantum attention (Level 3) |

```
Tier 0 (Exact)    ── ED, DMRG ──────────────────── "What is the right answer?"
Tier 1 (No attn)  ── RBM, CNN ──────────────────── "How well can simple NQS do?"
Tier 2 (Classical) ─ Simplified ViT, Classical ViT ─ "Does (data-dependent) attention help?"
Tier 3 (Quantum)  ── QSANN, QMSAN ─────────────── "Does quantum attention help more?"
```

---

## Project Structure

```
.
├── main.py                          # Thin CLI entry point
│
├── conf/                            # Composable YAML config tree
│   ├── config.yaml                  # Hydra root (defaults)
│   ├── solution/                    # One YAML per solution (model params)
│   │   ├── ed.yaml, dmrg.yaml
│   │   ├── rbm.yaml, cnn_resnet.yaml
│   │   ├── classical_vit.yaml, simplified_vit.yaml
│   │   └── qsann.yaml, qmsan.yaml
│   ├── hamiltonian/                 # Physics system definitions
│   │   ├── square_4x4.yaml         # 4×4 J1-J2 square, PBC
│   │   ├── square_6x6.yaml
│   │   └── chain_20.yaml           # 20-site 1D chain
│   ├── training/                    # Optimiser / VMC params
│   │   ├── default.yaml            # 500 steps, SR, 1024 samples
│   │   └── fast.yaml               # 20 steps, 128 samples
│   ├── evaluation/
│   │   └── default.yaml            # Which metrics to compute
│   └── experiment/                  # Task lists (like PTP presets)
│       ├── test.yaml               # Quick sanity (ED + fast RBM)
│       ├── ed.yaml                 # ED across all g values
│       ├── baseline.yaml           # All classical solutions
│       └── benchmark.yaml          # 6 solutions × 4 g = 24 tasks
│
├── src/
│   ├── experiment/                  # Experiment orchestration
│   │   ├── experiment.py            # Experiment class + config loading
│   │   └── task.py                  # Task runner (dispatch by solution)
│   │
│   ├── hamiltonians/                # Physics definitions
│   │   ├── lattice_utils.py         # Geometry: 1D chain, 2D square
│   │   ├── j1j2_chain.py           # 1D J1-J2 (QuSpin + NetKet)
│   │   └── j1j2_square.py          # 2D J1-J2 (QuSpin + NetKet)
│   │
│   ├── numerical_solvers/           # Tier 0: Reference solvers
│   │   ├── ed/
│   │   │   └── solver.py           # QuSpin Lanczos Exact Diagonalisation
│   │   └── dmrg/
│   │       └── solver.py           # TeNPy DMRG
│   │
│   ├── models/                      # NQS ansätze (Tiers 1-3)
│   │   ├── base_model.py           # BaseModel: x ∈ {±1}^N → log ψ(x)
│   │   ├── training/               # VMC training loop
│   │   │   ├── vmc_runner.py       # NetKet VMC driver wrapper
│   │   │   ├── sr_optimizer.py     # Stochastic Reconfiguration
│   │   │   └── callbacks.py        # Logging, checkpointing, early stop
│   │   ├── classical_models/       # Tier 1-2
│   │   │   ├── rbm/               # Restricted Boltzmann Machine (Tier 1)
│   │   │   │   └── model.py
│   │   │   ├── cnn_resnet/        # CNN + Residual Blocks (Tier 1)
│   │   │   │   └── model.py
│   │   │   ├── simplified_vit/    # Position-only attention, Level 0 (Tier 2)
│   │   │   │   └── model.py
│   │   │   └── classical_vit/     # Dot-product attention, Level 1 (Tier 2)
│   │   │       ├── attention.py   # Multi-head dot-product attention
│   │   │       ├── encoder.py     # Pre-LN transformer block
│   │   │       └── model.py
│   │   └── quantum_models/        # Tier 3
│   │       ├── base_quantum_model.py  # BaseQuantumModel(BaseModel)
│   │       │                          #   tokenise, angle_encoding,
│   │       │                          #   variational_layer, hva_layer, FFN
│   │       ├── qsann/            # Level 2: Gaussian-projected
│   │       │   ├── circuits.py    # Q/K/V PQC circuits
│   │       │   ├── attention.py   # GaussianProjectedAttention
│   │       │   └── model.py       # QSANN(BaseQuantumModel)
│   │       └── qmsan/            # Level 3: Swap-test
│   │           ├── circuits.py    # Swap test + value circuits
│   │           ├── attention.py   # SwapTestAttention
│   │           └── model.py       # QMSAN(BaseQuantumModel)
│   │
│   ├── evaluation/                  # Per-experiment metrics
│   │   ├── energy.py               # Relative error, E/N
│   │   ├── entanglement.py         # Von Neumann entropy via SVD
│   │   ├── param_efficiency.py     # Energy vs. parameter count
│   │   └── attention_viz.py        # Attention pattern visualisation
│   │
│   ├── utils/
│   │   ├── experiment.py           # Experiment directory creation
│   │   ├── mlflow_tracker.py       # MLflow integration
│   │   ├── logger.py               # Logging setup
│   │   └── constants.py            # Physical constants
│   │
│   └── paths.py                    # Project path definitions
│
├── scripts/                         # Cluster and utility scripts
│   ├── run_experiment.sbatch       # SLURM submission
│   ├── cleanup_failed.py           # Sync outputs/ ↔ mlruns/
│   ├── frequent_commands.md        # Copy-paste cheat sheet
│   └── nurc_cluster_commands.md    # Northeastern cluster setup
│
├── outputs/                         # Experiment results (gitignored)
│   └── {hamiltonian}/{method}_{timestamp}/
│       ├── energy_results.json
│       ├── convergence.csv
│       └── checkpoint.pkl
│
├── paper/                           # LaTeX manuscript
└── proposal.md                      # Research proposal + complexity analysis
```

---

## Architecture Deep Dive

### 1. Config System (`conf/`)

The config tree is **composable** — inspired by Hydra but loaded via plain YAML:

```
conf/experiment/test.yaml                  # EXPERIMENT PRESET
├── references → conf/solution/rbm.yaml    # SOLUTION (model + its params)
├── references → conf/hamiltonian/square_4x4.yaml  # PHYSICS SYSTEM
├── references → conf/training/fast.yaml   # TRAINING HYPERPARAMS
└── references → conf/evaluation/default.yaml      # EVAL METRICS
```

**How resolution works** (`src/experiment/experiment.py::resolve_task`):

A task entry like `{solution: rbm, hamiltonian: square_4x4, g: 0.5}` is resolved by:
1. Loading `conf/solution/rbm.yaml` → `resolved["solution"]`
2. Loading `conf/hamiltonian/square_4x4.yaml` → `resolved["hamiltonian"]`
3. Loading `conf/training/default.yaml` → `resolved["training"]`
4. Loading `conf/evaluation/default.yaml` → `resolved["evaluation"]`
5. Applying per-task overrides: `g: 0.5` → `resolved["hamiltonian"]["g"] = 0.5`

**To add a new solution**: create `conf/solution/my_model.yaml` with `name`, `type`, and model-specific params. Then add the type handler in `src/experiment/task.py::_build_model`.

**To add a new experiment**: create `conf/experiment/my_exp.yaml` with a `tasks:` list referencing existing solutions and hamiltonians.

### 2. Experiment Orchestration (`src/experiment/`)

```
main.py
  └── Experiment(conf_dir)           # experiment.py
        ├── add_tasks(tasks_list)    # from experiment YAML
        └── run()                    # sequential execution
              └── resolve_task()     # load component YAMLs
                    └── run_task()   # task.py — dispatch by type
                          ├── ed  → _run_ed()
                          ├── dmrg → _run_dmrg()
                          └── vmc → _run_vmc()
                                ├── _build_hamiltonian()
                                ├── _build_model()
                                └── train()  # vmc_runner.py
```

This mirrors the PTP `Experiment` class: `add_task` → `add_tasks` → `run` → `_print_summary`. The key difference is sequential execution (PTP uses async multi-queue for API/cluster providers; we just need sequential GPU).

### 3. BaseModel Interface (`src/models/base_model.py`)

**Every NQS model** subclasses `BaseModel`, a Flax `nn.Module`:

```python
class BaseModel(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x ∈ {+1, -1}^N → log ψ(x) ∈ ℂ"""
        raise NotImplementedError
```

This is the contract with NetKet's VMC driver. When you add a new model:
1. Subclass `BaseModel`
2. Implement `__call__`: take a spin config `(N,)`, return a complex scalar
3. Register in `src/experiment/task.py::_build_model`
4. Create `conf/solution/my_model.yaml`

### 4. Hamiltonian Construction (`src/hamiltonians/`)

Each Hamiltonian module exports **two builders**:

| Function | Returns | Used By |
|----------|---------|---------|
| `build_quspin_hamiltonian(...)` | QuSpin `hamiltonian` object | ED solver |
| `build_netket_hamiltonian(...)` | `(hilbert, H)` NetKet operator | VMC training |
| `hamiltonian_id(...)` | `str` like `"square_4x4_g0.5"` | Experiment dir naming |

**`lattice_utils.py`** provides geometry primitives:
- `build_chain_neighbors(L, pbc)` → list of `(i, j)` pairs for J1 and J2
- `build_square_neighbors(Lx, Ly, pbc)` → NN and NNN pairs for 2D

The J1-J2 Hamiltonian:
$$H = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j + J_2 \sum_{\langle\langle i,j \rangle\rangle} \vec{S}_i \cdot \vec{S}_j$$

where $g = J_2/J_1$ is the frustration ratio. The critical region $g \approx 0.5$ is where classical methods struggle.

### 5. Quantum Attention: QSANN vs QMSAN

The core research contribution. Both replace classical dot-product attention with PQC-based mechanisms:

#### QSANN (Level 2 — Gaussian-Projected)

```
spin config x → tokenise into (n_tokens, n_qubits) patches
  → Encode each token into PQC
    → Q circuit: angle_encode(x) → variational_layers → ⟨PauliZ⟩ per qubit → q_i ∈ ℝ^nq
    → K circuit: same but different params → k_j ∈ ℝ^nq
    → V circuit: same but different params → v_j ∈ ℝ^nq
  → Gaussian kernel: α_{ij} = softmax( -||q_i - k_j||² / 2σ² )
  → Output: Σ_j α_{ij} v_j
```

**Key insight**: Q and K are **classical vectors** (PauliZ expectations), but they're generated by quantum circuits that operate in 2^nq-dimensional Hilbert space. The similarity computation is classical (Gaussian kernel instead of dot product).

#### QMSAN (Level 3 — Swap Test)

```
spin config x → tokenise
  → Encode each token into PQC
    → Q circuit on register A (nq qubits)
    → K circuit on register B (nq qubits)
  → Swap test: ancilla + CSWAP(A, B) → ⟨Z_ancilla⟩ = |⟨ψ_q|ψ_k⟩|²
  → α_{ij} = softmax( |⟨q_i|k_j⟩|² )
  → V circuit: PauliZ expectations
  → Output: Σ_j α_{ij} v_j
```

**Key insight**: similarity is computed **entirely in Hilbert space** — no classical projection. This is the "fully quantum" attention. The swap test uses 2×nq + 1 qubits.

#### Complexity Comparison

| | QSANN | QMSAN | Classical ViT |
|--|-------|-------|---------------|
| Attention params | O(3·nq·nL) | O(3·nq·nL) | O(3·d²·L) |
| Forward (simulator) | O(n²·2^nq) | O(n²·2^{2nq}) | O(n²·d) |
| Forward (real QPU) | O(n²·nq·nL) | O(n²·nq·nL) | — |
| SR cost at train | O(P²) tiny | O(P²) tiny | O(P²) huge |

### 6. VMC Training Pipeline (`src/models/training/`)

```
vmc_runner.py::train(model, hilbert, H, experiment_dir, config, E_exact)
  │
  ├── 1. Build NetKet MetropolisExchange sampler
  ├── 2. Build SR optimizer (sr_optimizer.py)
  │       └── NetKet SR with diag_shift regularisation
  ├── 3. Build VMC driver (nk.VMC)
  ├── 4. Register callbacks (callbacks.py)
  │       ├── EnergyLogger — logs E, σ², E/N every N steps
  │       ├── EarlyStopping — stops if σ² < threshold for patience steps
  │       └── CheckpointSaver — saves params every N steps
  ├── 5. Run driver.run(n_steps)
  └── 6. Save results to experiment_dir/energy_results.json
```

**Stochastic Reconfiguration (SR)** is the key optimiser. It's natural gradient descent on the quantum geometric tensor:
$$\Delta\theta = -\eta \, S^{-1} \nabla_\theta \langle H \rangle$$

where $S_{ij} = \langle O_i^* O_j \rangle - \langle O_i^* \rangle \langle O_j \rangle$ is the Fisher information matrix. SR cost is O(P²) — this is why fewer parameters (quantum models) means cheaper training.

### 7. Evaluation (`src/evaluation/`)

Evaluation runs **per experiment** — each experiment directory gets its own metrics files:

- **`energy.py`**: Relative error $\epsilon = |E - E_\text{ED}| / |E_\text{ED}|$, energy per site $E/N$
- **`entanglement.py`**: Bipartite Von Neumann entropy $S = -\text{Tr}(\rho_A \log \rho_A)$ via SVD
- **`param_efficiency.py`**: Energy accuracy as function of parameter count
- **`attention_viz.py`**: Attention weight matrix visualisation

### 8. Experiment Directory Layout

Created by `src/utils/experiment.py::create_experiment_dir`:

```
outputs/
└── square_4x4_g0.5/                # hamiltonian identifier
    └── rbm_20260325_051400/         # method + timestamp
        ├── energy_results.json      # final E, σ², relative_error
        ├── convergence.csv          # step-by-step E, σ²
        └── checkpoint.pkl           # model parameters
```

MLflow runs are synced in `mlruns/`. The `cleanup_failed.py` script keeps them consistent.

---

## Quick Start

### Installation

```bash
git clone <repo>
cd qTransformer_on_frustrated_Heisenberg_model

conda create -n qml python=3.12
conda activate qml
pip install -e .

# Core dependencies
pip install netket pennylane pennylane-lightning jax jaxlib flax optax
pip install quspin physics-tenpy mlflow pyyaml
```

### Run Locally

```bash
# Quick sanity check (ED + fast RBM, ~30 seconds)
python main.py --batch --experiment test

# Show what a preset will do without running
python main.py --batch --experiment benchmark --dry-run

# Full classical baselines
python main.py --batch --experiment baseline

# Full benchmark (24 tasks — hours)
python main.py --batch --experiment benchmark
```

### Run on SLURM Cluster

```bash
ssh zhang.haoyu6@login.explorer.northeastern.edu
cd projects/qTransformer_on_frustrated_Heisenberg_model
module load anaconda3/2024.06 && source activate qml

# Submit experiment
sbatch scripts/run_experiment.sbatch test
sbatch scripts/run_experiment.sbatch benchmark

# Monitor
squeue -u $USER
tail -f logs/qtransformer_*.out
```

### Adding a New Model

1. **Create the model** in `src/models/classical_models/` or `src/models/quantum_models/`:
   ```python
   class MyModel(BaseModel):
       my_param: int = 32

       @nn.compact
       def __call__(self, x):
           # x: (N,) spin config → log ψ(x): complex scalar
           ...
           return log_psi
   ```

2. **Register it** in `src/experiment/task.py::_build_model`:
   ```python
   elif sol_type == "my_model":
       from src.models.classical_models.my_model import MyModel
       return MyModel(my_param=sol_cfg.get("my_param", 32))
   ```

3. **Create config** `conf/solution/my_model.yaml`:
   ```yaml
   name: my_model
   type: my_model
   my_param: 32
   ```

4. **Add to an experiment** `conf/experiment/test.yaml`:
   ```yaml
   tasks:
     - solution: my_model
       hamiltonian: square_4x4
       g: 0.0
   ```

### Adding a New Hamiltonian

1. **Create config** `conf/hamiltonian/my_system.yaml`:
   ```yaml
   name: my_system
   geometry: square    # or chain
   Lx: 8
   Ly: 8
   J1: 1.0
   g: 0.0
   pbc: true
   ```

2. Reference it in experiments: `hamiltonian: my_system`

---

## Key Design Decisions

### Lazy Imports
All heavy libraries (QuSpin, TeNPy, NetKet, PennyLane) are imported **inside functions**, not at module level. This keeps `import src` lightweight and avoids loading all backends when only one is needed.

### Physics-Centric Naming
Directory structure follows physics conventions:
- `outputs/{hamiltonian}/{method}` not `outputs/{model}/{dataset}`
- `conf/solution/` not `conf/ansatz/` (clearer to non-physics readers)
- `conf/hamiltonian/` describes the physical system, not a "dataset"

### Config Composition (not Inheritance)
Each experiment YAML defines a flat list of tasks. Each task references component configs by name. There is no YAML inheritance or override hierarchy — the `resolve_task` function handles merging. This is intentionally simple and explicit.

### Sequential Execution
Unlike PTP's async multi-queue (for API/cluster providers), this project runs tasks sequentially. VMC training is GPU-bound and single-process, so parallelism would not help within a single node.

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `jax`, `jaxlib` | ≥0.4 | Autodiff, JIT compilation |
| `flax` | ≥0.8 | Neural network modules |
| `optax` | ≥0.2 | Optimisers |
| `netket` | ≥3.10 | VMC engine, MCMC sampling, SR |
| `pennylane` | ≥0.35 | Quantum circuits |
| `pennylane-lightning` | ≥0.35 | C++ backend for PQC simulation |
| `quspin` | ≥1.0 | Exact diagonalisation |
| `physics-tenpy` | ≥1.0 | DMRG |
| `mlflow` | ≥2.0 | Experiment tracking |
| `pyyaml` | ≥6.0 | Config loading |

---

## References

1. Viteritti et al., "Transformer variational wave functions for frustrated quantum spin systems," arXiv (2023)
2. Rende et al., "Optimized attention mechanisms for variational wave functions on 2D J1-J2 Heisenberg model," arXiv (2025)
3. Li et al., "QSANN: Quantum Self-Attention Neural Network," Sci. China (2022)
4. Xu et al., "QMSAN: Quantum Mixed-State Self-Attention Network," arXiv (2024)
5. Evans et al., "SASQuaTCh: Variational Quantum Transformer with Kernel-Based Self-Attention," arXiv (2024)
6. Liu et al., "aCNN with residual blocks for frustrated Heisenberg J1-J2 model," arXiv (2024)
