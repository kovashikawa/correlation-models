"""Benchmark all measures on classic nonlinear counterexamples.

Prints a markdown table: rows are dependence measures, columns are
synthetic datasets. The story the table tells: Pearson is ~0 on
quadratic, |X|, sine, circle, and xor, while every modern measure
scores them as dependent.

Usage: uv run python scripts/benchmark.py
"""

from __future__ import annotations

import numpy as np

from correlation_models import measures as m

N = 10_000
SEED = 42


def make_datasets(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x = rng.normal(size=N)
    z = rng.normal(size=N)

    return {
        "linear": (x, 1.5 * x + rng.normal(scale=0.3, size=N)),
        "quadratic": (x, x**2),
        "abs": (x, np.abs(x)),
        "sine": (x, np.sin(4 * x)),
        "circle": (x, np.sqrt(np.maximum(1 - x**2, 0)) * rng.choice([-1.0, 1.0], size=N)),
        "xor": (x, np.where(x * z > 0, 1.0, -1.0)),
        "independent": (rng.normal(size=N), rng.normal(size=N)),
        "heavy_tail": (x, x + rng.standard_t(df=3, size=N) * 0.5),
    }


MEASURES = [
    ("Pearson", m.pearson),
    ("Spearman", m.spearman),
    ("Kendall", m.kendall),
    ("Chatterjee xi", m.chatterjee_xi),
    ("Distance corr", m.distance_correlation),
    ("HSIC", m.hsic),
    ("KSG MI", m.mutual_information_ksg),
    ("Tail dep (q=0.95)", lambda a, b: m.tail_dependence(a, b, q=0.95)),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    datasets = make_datasets(rng)

    print(f"n = {N}, seed = {SEED}\n")
    header = "| Measure | " + " | ".join(datasets.keys()) + " |"
    sep = "|---|" + "---|" * len(datasets)
    print(header)
    print(sep)

    for name, fn in MEASURES:
        row = [name]
        for x, y in datasets.values():
            row.append(f"{fn(x, y):.3f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
