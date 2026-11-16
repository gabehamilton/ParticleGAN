"""Generate the single current native-baseline leaderboard, ranked within served-law cohorts.

    python make_leaderboard.py RECORDS_ROOT --json-out PATH

RECORDS_ROOT is the local `outputs/native-baseline` directory (bulk receipts stay local). The script writes
`leaderboard.md` next to itself and the full per-config JSON to PATH (keep that local, out of Git).

This revises the original local `outputs/native-baseline/make_leaderboard.py` (sha256 ecd57848..., recorded in
provenance.json). Records, config identity and the ranking key are unchanged:
- Reads every RECORDS_ROOT/*/results.jsonl except quarantine/. A config is base preset + delta, tagged by source commit.
- Ranking: PASS on both selection tasks, ring8 terminal passes, gaussian1d terminal passes, fewest distinct ring8
  gates failing in the last 5 checks, smallest worst ring8 max_cov, then total passing checks.
What changed:
- Each config is placed in a served-law cohort read from its own receipt's resolved recipe (`continuous_policy`,
  `serve_average`; ring8 receipt first, else gaussian1d). Ranks are within a cohort. Cohorts are not comparable.
- A source of `?` means no readable receipt; it is shown as `? (no receipt)`.
- Nested earlier attempts (RECORDS_ROOT/*/attempt*/results.jsonl) are listed separately as off-board,
  non-numerical outcomes. They are neither ranked nor counted in the tuning budget.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECT = ("api-ring8-acquire", "api-gaussian1d-acquisition")
UNKNOWN = "unknown served law (no readable receipt recipe)"

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("root", type=Path, help="local outputs/native-baseline records directory")
parser.add_argument("--json-out", type=Path, required=True, help="local path for the full per-config JSON")
args = parser.parse_args()
ROOT = args.root.resolve()


def read_records(pattern):
    records = []
    for f in sorted(ROOT.glob(pattern)):
        if f.parent.name.startswith("quarantine"):
            continue
        seen = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            seen[(r["label"], r["case"])] = r  # last record wins (ERROR retries)
        for r in seen.values():
            r["dir"] = f.parent.relative_to(ROOT).as_posix()
            try:
                rec = json.loads(Path(r["receipt"]).read_text())
                r["source_commit"] = rec["source"].get("commit", "?")[:9]
                r["source_dirty"] = not rec.get("source_unchanged", True)
            except Exception:
                r["source_commit"], r["source_dirty"] = "?", None
            records.append(r)
    return records


def is_err(r):
    """Non-numerical outcome: run did not complete (receipt status ERROR/INVALID/INCOMPLETE/...). Never a FAIL."""
    return r is None or r.get("verdict") in ("ERROR", None) or r.get("status") not in (None, "COMPLETE")


def receipt_recipe(r):
    if not r or not r.get("receipt"):
        return None
    try:
        recipe = json.loads(Path(r["receipt"]).read_text()).get("recipe")
    except Exception:
        return None
    return recipe if isinstance(recipe, dict) else None


def cohort(c):
    """Served law the resolved recipe declares; samples are live and clean (no EMA, no output noise)."""
    recipe = next((found for found in (receipt_recipe(c["cases"].get(k)) for k in SELECT) if found), None)
    if recipe is None:
        return UNKNOWN
    policy, serve = recipe.get("continuous_policy"), recipe.get("serve_average") or 0
    if policy is None and not serve:
        return "pure-atom served law (no continuous policy; pure atoms on ring8)"
    law = f"continuous served law ({policy})" if policy is not None else "pure-atom served law"
    return law + (", state-selected serving (serve_average > 0)" if serve else "")


def last5(c, case):
    r = c["cases"].get(case)
    return -1 if is_err(r) else r.get("last5_passed", -1)


def rank(c):
    both = all(not is_err(c["cases"].get(k)) and c["cases"][k].get("verdict") == "PASS" for k in SELECT)
    checks = sum((c["cases"].get(k) or {}).get("checks_passed", 0) or 0 for k in SELECT)
    r8 = c["cases"].get(SELECT[0])
    gates = 99 if is_err(r8) else len(r8.get("last5_fail_counts") or {})
    cmax = 1e9 if is_err(r8) else (r8.get("worst_last5") or {}).get("max_cov_eigen<=1.5", 1e9)
    return (-both, -last5(c, SELECT[0]), -last5(c, SELECT[1]), gates, cmax, -checks)


def cell(r):
    if r is None:
        return "not run"
    if is_err(r):
        return f"{r.get('status') or 'ERROR'} (not a FAIL)"
    return f"{r['verdict']} {r['last5_passed']}/5 · {r['checks_passed']}/{r['checks']}"


def worst(r):
    if is_err(r):
        return ""
    w = r.get("worst_last5", {})
    short = {"hq>=0.9": "hq", "mass_tv<=0.075": "tv", "min_cov_eigen>=0.5": "cmin", "max_cov_eigen<=1.5": "cmax",
             "max_radial_ks<=0.1": "ks", "mean_error_sigma<=0.2": "mean", "cdf_ks<=0.05": "cdf",
             "std_ratio>=0.8": "std", "mean_error_sigma<=0.1": "mean", "radial_ks<=0.075": "rks",
             "max_projection_ks<=0.06": "pks", "min_cov_eigen>=0.85": "cmin", "max_cov_eigen<=1.15": "cmax"}
    return " ".join(f"{short[k]}={v}" for k, v in w.items() if k in short)


def source(c):
    return "? (no receipt)" if c["source"] == "?" else c["source"]


def delta_of(r):
    return r.get("fields") or r.get("overrides") or {}


configs = {}
for r in read_records("*/results.jsonl"):
    delta = delta_of(r)
    key = (r["recipe"], json.dumps(delta, sort_keys=True), r["source_commit"])
    c = configs.setdefault(key, {"base": r["recipe"], "delta": delta, "source": r["source_commit"],
                                 "labels": set(), "dirs": set(), "cases": {}})
    c["labels"].add(r["label"]); c["dirs"].add(r["dir"])
    prev = c["cases"].get(r["case"])
    if prev is None or is_err(prev):
        c["cases"][r["case"]] = r
for c in configs.values():
    c["cohort"] = cohort(c)

rows = sorted(configs.values(), key=rank)
cohorts = sorted({c["cohort"] for c in rows}, key=lambda name: (name == UNKNOWN, name.startswith("continuous"), name))
holdout_cases = sorted({k for c in rows for k in c["cases"] if k not in SELECT})
distinct = lambda group: len({(c["base"], json.dumps(c["delta"], sort_keys=True)) for c in group})
lines = ["# Native baseline leaderboard (current)", "",
         "Generated by [`make_leaderboard.py`](make_leaderboard.py) from every sweep record under the local "
         "`outputs/native-baseline/` (quarantined dirty-tree runs excluded).",
         "Selection tasks: `api-ring8-acquire`, `api-gaussian1d-acquisition`. Protocol seed 0; case-owned budgets and cadence.",
         "A cell reads `verdict terminal-passes/5 · passing-checks/24`. Worst-of-last-5 metrics follow.",
         "A non-COMPLETE receipt status (ERROR/INVALID/INCOMPLETE) is shown as such; it is not a numerical FAIL.",
         "Source `? (no receipt)` means the run left no readable receipt.", "",
         "**Rows are ranked within served-law cohorts.** Each config's cohort comes from its own receipt's resolved "
         "recipe (`continuous_policy`, `serve_average`). Served samples are live and clean. Different served laws are "
         "separate cohorts, so do not compare ranks or metrics across sections.", "",
         f"Distinct trainer configurations measured: **{distinct(rows)}** "
         "(disclosed tuning budget; a config re-measured on another source commit is listed per source but counted once)."]
placements = {}
for c in rows:
    placements.setdefault((c["base"], json.dumps(c["delta"], sort_keys=True)), set()).add(c["cohort"])
split = {key: names for key, names in placements.items() if len(names) > 1}
if split:
    lines.append(f"Per-cohort counts sum to {sum(map(len, placements.values()))}, because {len(split)} configs have "
                 "re-measurements in more than one cohort: " + "; ".join(
                     f"`{base}` + `{delta}` in " + " and ".join(sorted(names)) for (base, delta), names in sorted(split.items()))
                 + ".")
header = ("| # | base + delta | source | ring8 | ring8 worst-last-5 | gaussian1d | g1d worst-last-5 |" +
          "".join(f" {k} |" for k in holdout_cases))
rule = "|---|---|---|---|---|---|---|" + "---|" * len(holdout_cases)
for name in cohorts:
    group = [c for c in rows if c["cohort"] == name]
    lines += ["", f"## {name[0].upper() + name[1:]}", "",
              f"{distinct(group)} distinct configurations ({len(group)} rows).", "", header, rule]
    for i, c in enumerate(group, 1):
        r8, g1 = c["cases"].get(SELECT[0]), c["cases"].get(SELECT[1])
        hold = " | ".join(cell(c["cases"].get(k)) + (" " + worst(c["cases"].get(k)) if c["cases"].get(k) else "")
                          for k in holdout_cases)
        lines.append(f"| {i} | `{c['base']}` + `{json.dumps(c['delta'], sort_keys=True)}` | {source(c)} | {cell(r8)} | "
                     f"{worst(r8)} | {cell(g1)} | {worst(g1)} |" + (f" {hold} |" if holdout_cases else ""))

attempts = {}
for r in read_records("*/attempt*/results.jsonl"):
    attempt = attempts.setdefault(r["dir"], {"labels": set(), "configs": set(), "cases": set(), "statuses": set()})
    attempt["labels"].add(r["label"]); attempt["cases"].add(r["case"])
    attempt["configs"].add((r["recipe"], json.dumps(delta_of(r), sort_keys=True)))
    attempt["statuses"].add("COMPLETE" if not is_err(r) else (r.get("status") or "ERROR"))
if attempts:
    lines += ["", "## Off-board earlier attempts (non-numerical, not ranked, not in the tuning budget)", "",
              "Superseded attempt directories nested under a sweep directory. None produced a numerical verdict.", "",
              "| attempt | labels | recorded configs | statuses |", "|---|---|---|---|"]
    for name, attempt in sorted(attempts.items()):
        recorded = "; ".join(f"`{base}` + `{delta}`" for base, delta in sorted(attempt["configs"]))
        lines.append(f"| `{name}` | {', '.join(sorted(attempt['labels']))} | {recorded} | "
                     f"{', '.join(sorted(attempt['statuses']))} |")

(HERE / "leaderboard.md").write_text("\n".join(lines) + "\n")
args.json_out.parent.mkdir(parents=True, exist_ok=True)
args.json_out.write_text(json.dumps([{**{k: v for k, v in c.items() if k not in ("labels", "dirs", "cases")},
                                      "labels": sorted(c["labels"]), "dirs": sorted(c["dirs"]),
                                      "cases": {k: {kk: vv for kk, vv in v.items() if kk != "error"}
                                                for k, v in c["cases"].items()}} for c in rows], indent=1))
print(f"{len(rows)} rows, {distinct(rows)} distinct configs in {len(cohorts)} cohorts; "
      f"{len(attempts)} off-board attempt dirs")
for name in cohorts:
    group = [c for c in rows if c["cohort"] == name]
    top = group[0]
    print(f"  {name}: {distinct(group)} configs; top {top['base']} -> "
          f"{cell(top['cases'].get(SELECT[0]))} / {cell(top['cases'].get(SELECT[1]))}")
