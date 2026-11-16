"""ring8 gate calibration witnesses (EVALUATOR CALIBRATION ONLY; never a trained-recipe result).

python witness.py REPO   -> prints one JSON line per witness; scored by REPO's definition_quality.ring_metrics
with the harness draw law (4096 uniform row draws from 256 equal atoms; eval seeds 10000..10099).
Free atoms are optimized directly in output space with no generator, critic or ParticleGAN recipe.
"""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from benchmarks.toy_audit.definition_quality import ring_metrics

torch.set_num_threads(1)
ang = torch.arange(8) * 2 * math.pi / 8
C = 3 * torch.stack([ang.cos(), ang.sin()], 1)
SEEDS = range(10000, 10100)


def target(n, g):
    return C[torch.randint(0, 8, (n,), generator=g)] + torch.randn(n, 2, generator=g) * .07


def evaluate(atoms, blur=0.):
    rows = []
    for s in SEEDS:
        g = torch.Generator().manual_seed(s)
        pts = atoms[torch.randint(0, len(atoms), (4096,), generator=g)]
        if blur:
            pts = pts + blur * .07 * torch.randn(4096, 2, generator=g)
        m = ring_metrics(pts.numpy()); e = np.array(m["covariance_eigenvalues"])
        rows.append((m["passed"], m["max_radial_ks"], m["hq"], e.min(), e.max(), m["mass_tv"]))
    a = np.array(rows, dtype=float)
    return dict(pass_rate=a[:, 0].mean(), median_ks=np.median(a[:, 1]), median_hq=np.median(a[:, 2]),
                median_cmin=np.median(a[:, 3]), median_cmax=np.median(a[:, 4]), median_tv=np.median(a[:, 5]))


def dist(a, b):
    return (torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist").square() + 1e-12).sqrt()


def sliced(x, y, P=32):
    a = torch.arange(P) * (math.pi / P); F = torch.stack((a.cos(), a.sin()))
    px, py = (x @ F).sort(0).values, (y @ F).sort(0).values
    return (px - py[((torch.arange(len(x)) + .5) / len(x) * len(y)).long()]).square().mean()


def optimize(loss_kind, memory=4096, blur=0., steps=1200):
    g = torch.Generator().manual_seed(1)
    x = (C[torch.arange(256) % 8] + torch.randn(256, 2, generator=g) * .3).requires_grad_()
    opt = torch.optim.Adam([x], lr=3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    mem = target(memory, g)
    for _ in range(steps):
        xs = x + blur * .07 * torch.randn(256, 2, generator=g) if blur else x
        loss = 2 * dist(xs, mem).mean() - dist(xs, xs).mean() if loss_kind == "energy" else sliced(xs, mem)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        mem = torch.cat([mem[128:], target(128, g)])  # FIFO of seen samples, 128 per update
    return x.detach()


k = 32
r = np.sqrt(-2 * np.log(1 - (np.arange(k) + .5) / k)); th = np.arange(k) * math.pi * (3 - math.sqrt(5))
strat = torch.tensor(np.stack([r * np.cos(th), r * np.sin(th)], 1) * .07, dtype=torch.float32)
out = {"stratified_32_per_mode": evaluate(torch.cat([C[i] + strat for i in range(8)]))}
rng = torch.Generator().manual_seed(0)
out["iid_32_per_mode"] = evaluate(torch.cat([C[i] + torch.randn(k, 2, generator=rng) * .07 for i in range(8)]))
out["free_atoms_sliced_w2_memory4096"] = evaluate(optimize("sliced"))
out["free_atoms_energy_memory4096"] = evaluate(optimize("energy"))
out["free_atoms_energy_memory4096_blur0.5"] = evaluate(optimize("energy", blur=.5), blur=.5)
for name, v in out.items():
    print(json.dumps({"witness": name, **{kk: round(float(vv), 4) for kk, vv in v.items()}}))
