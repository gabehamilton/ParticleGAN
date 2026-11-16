"""ring8 stray-atom witness (EVALUATOR CALIBRATION ONLY; never a trained-recipe result).

python stray_witness.py REPO   -> prints one JSON line per stray distance; scored by REPO's
definition_quality.ring_metrics with the witness.py draw law (4096 uniform row draws from 256 equal atoms;
eval seeds 10000..10099). It starts from witness.py's passing hand-stratified set (32 atoms per mode) and
moves one atom of mode 0 radially outward to k * sigma (sigma = .07) from its centre. It reports how far a
single stray must sit before the max covariance gate (eigenvalue / sigma^2 <= 1.5) fails.
"""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from benchmarks.toy_audit.definition_quality import ring_metrics

torch.set_num_threads(1)
SIGMA = .07
ang = torch.arange(8) * 2 * math.pi / 8
C = 3 * torch.stack([ang.cos(), ang.sin()], 1)
SEEDS = range(10000, 10100)

k = 32
r = np.sqrt(-2 * np.log(1 - (np.arange(k) + .5) / k)); th = np.arange(k) * math.pi * (3 - math.sqrt(5))
strat = torch.tensor(np.stack([r * np.cos(th), r * np.sin(th)], 1) * SIGMA, dtype=torch.float32)
base = torch.cat([C[i] + strat for i in range(8)])
outward = C[0] / C[0].norm()

for distance in (0, 5, 6, 7, 8):
    atoms = base.clone()
    if distance:
        atoms[0] = C[0] + distance * SIGMA * outward
    rows = []
    for s in SEEDS:
        g = torch.Generator().manual_seed(s)
        m = ring_metrics(atoms[torch.randint(0, len(atoms), (4096,), generator=g)].numpy())
        e = np.array(m["covariance_eigenvalues"])
        rows.append((m["passed"], e.max(), m["max_radial_ks"], m["hq"]))
    a = np.array(rows, dtype=float)
    print(json.dumps({"witness": f"stratified_32_per_mode_one_atom_at_{distance}_sigma",
                      "pass_rate": round(float(a[:, 0].mean()), 4), "median_cmax": round(float(np.median(a[:, 1])), 4),
                      "median_ks": round(float(np.median(a[:, 2])), 4), "median_hq": round(float(np.median(a[:, 3])), 4)}))
