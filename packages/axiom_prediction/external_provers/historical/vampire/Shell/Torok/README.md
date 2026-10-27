# Torok guidance

Torok keeps every input clause and samples passive clauses proportionally to
its learned weights. This matches Connections' SATResetCoP **Weighted** mode
(temperature 1, probability floor `1e-6`), adapted to Vampire's given-clause loop.
It does not apply a top-k cutoff or rescore derived clauses with the network.

There is one new prover option: `--torok_file MODEL`. An empty path (the default)
disables it. The existing `--sine_selection` option is unchanged. Enabling both
scores the clauses remaining after SInE. Experimental support and filtering modes
and their equal-budget comparisons are described in `experiments/README.md`.

## Build and export

Use a LibTorch installation compatible with the Python PyTorch used for export.
CPU inference is used, with one inference thread. A CPU-only LibTorch package
is sufficient; a CUDA-enabled installation may also require its CUDA build tools.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build --target vampire --parallel 18
python Shell/Torok/export.py torok.pt Shell/Torok/torok.ts --connections ../connections
```

`torok.pt` is the canonical Connections training checkpoint. `Shell/Torok/torok.ts` is its
exported inference artifact; pass **the exported artifact** to Vampire. The
exporter copies trained layers into a tensor-only module and supports the Full
ReLU architecture. It uses TorchScript, a deprecated but available LibTorch
loading interface; exporting/loading has been tested with the installed PyTorch
2.14.1. Prefer Python 3.12/3.13 for export: PyTorch warns about TorchScript under
Python 3.14. Python is not needed at proof-search runtime.

Every build includes Torok and requires LibTorch. The model is loaded lazily
when `--torok_file MODEL` is used and reused within a process.

## Compare

Run from the repository root, replacing `problem.p` with the same input in each
command. Use a nonzero seed: Vampire treats seed zero as nondeterministic.

```bash
build/vampire --mode vampire --saturation_algorithm otter --random_seed 1 \
  --time_limit 30s --torok_file Shell/Torok/torok.ts problem.p
build/vampire --mode vampire --saturation_algorithm otter --random_seed 1 \
  --time_limit 30s --sine_selection axioms problem.p
build/vampire --mode vampire --saturation_algorithm otter --random_seed 1 \
  --time_limit 30s --sine_selection off problem.p
```

Keep the strategy, inputs, time limits and seeds fixed across methods; repeat
with several seeds. Charge graph construction and model inference to the run
budget, and also record process wall time if charging library/startup overhead.
`--show_preprocessing on` reports Torok's scored/retained counts and graph plus
inference time. Normal proof output and statistics remain available.

This compares different interventions: SInE filters formulas before
clausification; Torok scores clauses after clausification and changes passive
selection. Torok replaces age/weight and split queues. LRS age/weight pruning is
inactive for this queue, so use Otter for a straightforward controlled comparison.
Normal logical simplification, redundancy elimination and resource limits still
apply. Start with single-strategy `--mode vampire`, not a portfolio whose slices
may enable incompatible SInE settings or other saturation algorithms.

## Representation and inherited weights

The graph uses the same features and relations as Connections: ground terms are
shared globally, variables and non-ground terms are clause-local, and complement
edges require opposite signs and ordered-argument unification with variables
renamed apart. Symbol names carry no learned embedding. Equality argument order
and clauses can differ between the two provers' preprocessing pipelines; graph
parity is tested on shared CNF, not asserted for independently clausified FOF.
Connections may also add explicit equality axioms during FOF clausification,
whereas Vampire handles equality natively. The initial implementation supports
untyped first-order inputs, including equality.

Input conjecture clauses have weight 1. Other input clauses receive the model's
sigmoid score, floored at `1e-6`. A derived clause inherits the geometric mean
of its immediate parents' weights. A parentless introduced unit receives the
geometric mean of all scored input weights, including conjecture weights.
This inheritance rule is a Vampire-specific heuristic, not something learned by
the original checkpoint. Weights are cached by unit number for the current problem.

The passive queue uses exponential clocks to sample proportionally to weights
without scanning every clause for every draw. It uses Vampire's existing random
seed. This has the same categorical sampling law as Connections' Gumbel-based
choice, but does not reproduce its random sequence or its proof-search algorithm.
No clauses are discarded by Torok, and proof ancestry is unchanged.

## Verify

```bash
cmake --build build --target torok_check --parallel "$(nproc)"
python Shell/Torok/verify.py --connections ../connections \
  --checkpoint torok.pt --exported Shell/Torok/torok.ts --checker build/torok_check
```

This checks native graph tables and probabilities against Python on shared CNF,
including empty node groups, variable sharing and occurs-check complements. The
native checker also exercises inherited weights and weighted queue sampling.

## Compatible models

`--torok_file` accepts a CPU-compatible TorchScript module with
`forward(nodes, edges, axioms, conjectures)`. The two lists contain int64 graph
feature/edge tables in the order defined in `Model.cpp`; the final two int64
vectors index the axiom and conjecture clauses. The model must preserve these
feature meanings and return a finite float32 vector with one logit per axiom,
in the supplied axiom order. Vampire applies sigmoid itself.

Hidden dimensions, layer counts and architecture may differ. Matching tensor
sizes alone is insufficient if the graph encoding or output meanings differ.
Connections' new `.ts` checkpoints implement this interface directly.
