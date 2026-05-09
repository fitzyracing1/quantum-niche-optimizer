# quantum-niche-optimizer

QAOA quantum optimization for finding the best product niches to focus on.
Runs on **IBM Quantum real hardware** or a local **Aer simulator** (no account needed).

Handles arbitrarily large niche sets via **hierarchical chunking** — splits into
groups of 25 qubits, runs QAOA on each chunk, then runs a final QAOA pass on
the winners to find the global optimum.

## Install

```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

## Usage

```bash
# Local simulator (no account needed)
python quantum_optimizer.py

# Real IBM Quantum hardware (free account at quantum.ibm.com)
export IBM_QUANTUM_TOKEN=your_token
python quantum_optimizer.py --real

# Custom budget (how many niches to pick)
python quantum_optimizer.py --budget 10
```

## As a library

```python
from quantum_optimizer import optimize_strategy, mcp_quantum_optimize, mcp_quantum_random

# Find optimal 5 niches from your product catalog
result = optimize_strategy(budget=5)
print(result["recommended_niches"])

# Quantum random bits
bits = mcp_quantum_random(count=16)
print(bits["integer"])
```

## How it works

1. Loads all product metadata from your catalog
2. Builds a cost Hamiltonian encoding revenue weights and audience overlap penalties
3. Runs QAOA (Quantum Approximate Optimization Algorithm) to find the subset
   that maximizes revenue while minimizing cannibalization between niches
4. Hierarchical chunking handles catalogs larger than 25 niches by:
   - Round 1: QAOA on chunks of 25 niches each
   - Round 2: Final QAOA pass on all chunk winners

## Output

```json
{
  "recommended_niches": ["email inbox parser", "batch image resizer", ...],
  "score": 145.0,
  "method": "Hierarchical-QAOA-Aer",
  "rounds": 2,
  "total_niches_evaluated": 90,
  "budget": 5
}
```

## Part of pyautotools

- PyPI: [pyautotools](https://pypi.org/project/pyautotools/)
- Store: [fitzyracing1.github.io/tools](https://fitzyracing1.github.io/tools/)
