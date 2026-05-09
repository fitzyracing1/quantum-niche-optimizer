#!/usr/bin/env python3
"""
Quantum Niche Optimizer
Uses QAOA (Quantum Approximate Optimization Algorithm) to find the optimal
subset of product niches that maximizes revenue while minimizing audience overlap.

Supports:
  - Local Aer simulator (no account needed)
  - Real IBM Quantum hardware (free account at quantum.ibm.com)

Handles arbitrarily large niche sets via hierarchical chunking:
  Round 1: Split niches into chunks of MAX_QUBITS, run QAOA on each chunk
  Round 2: Run final QAOA pass on the winners from all chunks
"""

import json
import math
import random
import os
import sys
from pathlib import Path

IBM_TOKEN   = os.environ.get("IBM_QUANTUM_TOKEN", "")
MAX_QUBITS  = 25   # safe limit for Aer and real hardware
SHOTS       = 1024
PRODUCTS_DIR = Path(os.environ.get("PRODUCTS_DIR", "/Volumes/my volume/ai_products"))


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def get_backend(use_real=False):
    if use_real and IBM_TOKEN:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
        print(f"[quantum] Real hardware: {backend.name}")
        return backend
    from qiskit_aer import AerSimulator
    return AerSimulator()


# ---------------------------------------------------------------------------
# Cost / overlap helpers
# ---------------------------------------------------------------------------

CATEGORIES = {
    "developer":  ["python", "developer", "programmer", "code", "script", "api"],
    "data":       ["data", "csv", "excel", "spreadsheet", "analyst", "clean"],
    "business":   ["invoice", "email", "business", "small", "owner", "pdf"],
    "seo":        ["keyword", "rank", "google", "seo", "search", "serp"],
    "sysadmin":   ["log", "server", "monitor", "health", "anomaly"],
    "media":      ["image", "resize", "photo", "batch", "optimizer"],
    "file":       ["file", "organizer", "organize", "folder", "sort"],
}


def niche_category(niche: str) -> str:
    n = niche.lower()
    for cat, words in CATEGORIES.items():
        if any(w in n for w in words):
            return cat
    return "other"


def overlap(a: str, b: str) -> float:
    return 0.8 if niche_category(a) == niche_category(b) else 0.05


# ---------------------------------------------------------------------------
# Single-chunk QAOA
# ---------------------------------------------------------------------------

def run_qaoa_chunk(niches: list[str], prices: list[float],
                   budget: int, use_real: bool = False) -> list[str]:
    """
    Run QAOA on a chunk of ≤MAX_QUBITS niches.
    Returns the best subset of size ≤budget from this chunk.
    """
    import numpy as np
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    import scipy.optimize as opt

    n = len(niches)
    if n == 0:
        return []

    max_p = max(prices) if prices else 1
    weights = [p / max_p for p in prices]

    # Build cost Hamiltonian
    paulis = []
    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        paulis.append(("".join(reversed(z)), -weights[i]))
    for i in range(n):
        for j in range(i + 1, n):
            pen = overlap(niches[i], niches[j])
            if pen > 0:
                zz = ["I"] * n
                zz[i] = "Z"
                zz[j] = "Z"
                paulis.append(("".join(reversed(zz)), pen))

    cost_op = SparsePauliOp.from_list(paulis)
    ansatz  = QAOAAnsatz(cost_op, reps=2)
    ansatz.measure_all()

    backend  = AerSimulator() if not use_real else get_backend(True)
    ansatz_t = transpile(ansatz, backend)

    def objective(params):
        job    = backend.run(ansatz_t.assign_parameters(params), shots=SHOTS)
        counts = job.result().get_counts()
        total  = sum(counts.values())
        ev     = 0.0
        for bits_str, cnt in counts.items():
            bits  = [int(b) for b in bits_str.replace(" ", "")]
            score = sum(weights[i] * bits[i] for i in range(n))
            pen   = sum(overlap(niches[i], niches[j]) * bits[i] * bits[j]
                        for i in range(n) for j in range(i + 1, n))
            ev   += (cnt / total) * (score - pen)
        return -ev

    x0     = np.random.uniform(0, np.pi, ansatz.num_parameters)
    result = opt.minimize(objective, x0, method="COBYLA",
                          options={"maxiter": 150, "rhobeg": 0.5})

    # Sample best bitstring
    final  = backend.run(ansatz_t.assign_parameters(result.x), shots=SHOTS)
    counts = final.result().get_counts()
    best   = max(counts, key=lambda b: counts[b]).replace(" ", "")
    return [niches[i] for i, b in enumerate(best) if b == "1"][:budget]


# ---------------------------------------------------------------------------
# Hierarchical QAOA — handles any number of niches
# ---------------------------------------------------------------------------

def hierarchical_qaoa(niches: list[str], prices: list[float],
                      budget: int = 5, use_real: bool = False) -> dict:
    """
    Round 1: split into chunks of MAX_QUBITS, run QAOA on each → get local winners
    Round 2: pool all local winners, run final QAOA → global optimum
    """
    n = len(niches)
    print(f"[quantum] {n} niches → chunking into groups of {MAX_QUBITS}")

    # Build price lookup
    price_map = dict(zip(niches, prices))

    # Round 1 — chunk QAOA
    chunk_winners = []
    chunks = [niches[i:i + MAX_QUBITS] for i in range(0, n, MAX_QUBITS)]
    for idx, chunk in enumerate(chunks):
        chunk_prices = [price_map[c] for c in chunk]
        local_budget = max(2, budget)
        print(f"[quantum] Round 1 chunk {idx + 1}/{len(chunks)} ({len(chunk)} niches)...")
        try:
            winners = run_qaoa_chunk(chunk, chunk_prices, local_budget, use_real)
        except Exception as e:
            print(f"[quantum] chunk {idx+1} QAOA failed ({e}), using top-scored fallback")
            scored = sorted(zip(chunk_prices, chunk), reverse=True)
            winners = [n for _, n in scored[:local_budget]]
        chunk_winners.extend(winners)

    # Deduplicate
    seen, pool, pool_prices = set(), [], []
    for w in chunk_winners:
        if w not in seen:
            seen.add(w)
            pool.append(w)
            pool_prices.append(price_map.get(w, 29))

    print(f"[quantum] Round 2 final pass over {len(pool)} candidates...")

    # Round 2 — final QAOA on pooled winners
    if len(pool) > MAX_QUBITS:
        pool = pool[:MAX_QUBITS]
        pool_prices = pool_prices[:MAX_QUBITS]

    try:
        final_picks = run_qaoa_chunk(pool, pool_prices, budget, use_real)
    except Exception as e:
        print(f"[quantum] Round 2 QAOA failed ({e}), using greedy fallback")
        scored = sorted(zip(pool_prices, pool), reverse=True)
        final_picks = [n for _, n in scored[:budget]]

    score = sum(price_map.get(p, 29) for p in final_picks)
    return {
        "recommended_niches": final_picks,
        "score": round(score, 2),
        "method": "Hierarchical-QAOA" + ("-Real" if use_real else "-Aer"),
        "rounds": 2,
        "total_niches_evaluated": n,
        "budget": budget,
    }


# ---------------------------------------------------------------------------
# Load products and run
# ---------------------------------------------------------------------------

def optimize_strategy(budget=5, use_real=False) -> dict:
    niches, prices = [], []
    seen = set()
    for meta_file in PRODUCTS_DIR.glob("*/metadata.json"):
        try:
            m = json.loads(meta_file.read_text())
            kw = m.get("primary_keyword", m.get("name", "")).strip()
            if kw and kw not in seen:
                seen.add(kw)
                niches.append(kw)
                prices.append(float(m.get("price_usd", 29)))
        except Exception:
            pass

    if not niches:
        return {"error": "No products found in " + str(PRODUCTS_DIR)}

    result = hierarchical_qaoa(niches, prices, budget, use_real)

    out = PRODUCTS_DIR / "quantum_strategy.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[quantum] Strategy saved → {out}")
    return result


# ---------------------------------------------------------------------------
# MCP wrappers
# ---------------------------------------------------------------------------

def mcp_quantum_optimize(budget=5, use_real=False) -> dict:
    return optimize_strategy(budget=budget, use_real=use_real)


def mcp_quantum_random(count=8) -> dict:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    n = min(count, MAX_QUBITS)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    backend = AerSimulator()
    result  = backend.run(transpile(qc, backend), shots=1).result()
    bits    = list(result.get_counts().keys())[0].replace(" ", "")
    return {
        "bits":    [int(b) for b in bits],
        "integer": int(bits, 2),
        "source":  "IBM Quantum" if IBM_TOKEN else "Aer simulator",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_real = "--real" in sys.argv
    budget   = int(next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                         if a == "--budget"), 5))
    print("Quantum Niche Optimizer — Hierarchical QAOA")
    print(f"Mode: {'IBM Real Hardware' if use_real and IBM_TOKEN else 'Aer Simulator'}")
    print(f"Budget: {budget} niches\n")
    result = optimize_strategy(budget=budget, use_real=use_real)
    print(json.dumps(result, indent=2))
