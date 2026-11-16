# Native baseline: energy transport over seen-data memory (2026-11-13)

**Verdict: faithful numerical FAIL on ring8, PASS on gaussian1d.** No configuration among the
162 measured selection configurations passes both selection tasks. The selected, best-failing configuration is
registered as the named preset `k3p_energy`. It records a declared-delta native baseline. It is **not** a default,
has **no** calibration acceptance and is **not** independently confirmed.

[Single current leaderboard for this goal](leaderboard.md) · [Provenance (local receipt paths and sha256)](provenance.json) ·
[Calibration witnesses](calibration/) · [Mechanism code](../../../particlegan/energy_transport.py) ·
[API docs, `k3p_energy` delta table](../../../docs/api.md) · [Forge technique inventory](../technique-inventory.md)

## Question

Can one global public ParticleGAN trainer configuration pass the sustained gate on both selection tasks?
The gate requires all of the last 5 of 24 post-update checks to pass. Hosts, budgets, data law, batch sequence,
initialization, prior capacity, sampling and evaluation cadence are fixed by the task. Protocol seed 0 only, with no
seed variation.

| selection task | budget | prior / served law | gates |
|---|---|---|---|
| `api-ring8-acquire` | 1200 updates | 256-row learnable particle table, 4-D latent, pure atoms (sigma_rel 0); G is an MLP 4→96×3→2 | 4096 live draws (no EMA), nearest of 8 centres (radius 3, σ=.07): modes = 8, hq ≥ .90, mass_tv ≤ .075, per-mode covariance eigenvalues/σ² in [.5, 1.5], per-mode radial KS ≤ .10 |
| `api-gaussian1d-acquisition` | 1000 updates | MoG prior, 256 learnable means, σ = .025, z_dim 2 | target N(2, .5²): mean_error_sigma ≤ .2, std_ratio in [.8, 1.2], cdf_ks ≤ .05, finite, n ≥ 4096 |

Holdouts were never used for selection: `api-gaussian2d`, `api-ring8-hold`, `api-ring8-shift` and all ring16 cases.

## Selected configuration (`k3p_energy`)

`k3p` + {`energy_transport_weight` 300, `energy_transport_memory` 4096, `energy_transport_population` true,
`output_noise_std` 0, `lr_anneal_start` .2, `lr_floor` .01, `network_lr_floor` .001}.
The last four fields are the round-1 schedule delta. The three energy fields are new, technique-owned Recipe fields.

| task | verdict | worst of last 5 checks |
|---|---|---|
| ring8 | **FAIL 0/5** (0/24 checks) | hq .968, mass_tv .0168, cov eigen .61..**2.30**, radial KS **.201**. Only `max_cov_eigen ≤ 1.5` and `max_radial_ks ≤ .10` fail |
| gaussian1d | **PASS 5/5** (22/24 checks) | mean_err .036σ, std_ratio .969, cdf_ks .023 |

Against the round-1 best (k3p + kinetic transport 2 / local 1 + schedule; hq .957, tv .026, cov .56..26.0, KS .306),
energy cuts max_cov from 26.0 to 2.30 and KS from .306 to .201 at equal hq. That is the whole gain, and it is not a pass.

**Evidence identity.** The same numbers were measured at five sources:
- the sweep (ed3a393a5, label `k3p` + delta);
- the confirmation reruns with the named preset (a85a34fb1);
- the first port onto develop 805713ed5 (16dfd3982);
- the ported library head, with pinned sources restored (41b2acb2c);
- this branch's review-fix library head (859c70e1d).

All 25 observations, `observations.npz`, the batch sequence and the initial models are identical in every case.
On every develop-based head, `initial_state_sha256` and `final-state.pt` are also byte-identical to the
a85a34fb1 confirmation. These reruns re-identify one result; they are not new evidence.

## Calibration witnesses (evaluator calibration only, never trained results)

`calibration/witness.py` prints one JSON line per witness; `calibration/witness.json` holds those five rows. It scores hand-built and free-atom point sets with the harness draw law
(4096 uniform draws from 256 equal atoms) over eval seeds 10000-10099. The free atoms are optimized directly in output space, with no G, critic or recipe.

| witness | pass rate | median radial KS | median hq | median cov min..max |
|---|---|---|---|---|
| hand-stratified, 32 atoms/mode (radial quantiles) | .99 | .066 | 1.00 | .81..1.20 |
| ideal iid, 32 atoms/mode | 0 | .208 | .969 | .44..1.78 |
| free atoms, sliced W2 vs 4096 FIFO | 0 | .228 | .824 | 1.55..11.9 |
| free atoms, energy distance vs 4096 FIFO | 0 | .130 | .981 | .76..1.84 |
| free atoms, energy + 0.5σ continuous spread (trained and served) | 0 | .071 | .982 | .82..4.37 |

At N = 256 pure atoms, ring8 is a quantization test. The atoms must be placed better than iid, and there can be no strays.
The hand-stratified row shows that a pure-atom law with good placement passes: N = 256 is not too small for the gate.
`calibration/stray_witness.py` (output in `calibration/stray_witness.json`, same draw law and seeds) moves one atom of
that passing set radially away from its centre. The median max_cov is 1.78 at 5σ (pass rate .09), 2.11 at 6σ, 2.51 at 7σ
and 2.96 at 8σ, against the 1.5 gate. A single stray at 5σ already fails the gate on most draws.

## Mechanism

The energy distance is E = (2·mean ρ(x−y) − mean ρ(x−x′)) / s. It is computed from one generated draw per prior row (the
whole 256-atom population) to a FIFO of the 4096 most recently seen G-phase real rows. Here ρ is a smoothed Euclidean
norm and s is the detached target scale. E is an exact, repulsive transport objective on the G/prior loss, owned by the recipe.
- It fixes mass allocation: mass_tv falls from the round-1 locked .16 to .017.
- In the selected run (weight 300, memory 4096), transit atoms are gone from the final draws: 0 of 4096 samples lie more
  than .5 from a centre, against 6.3% with energy off. This is measured for that run only. Other high weights still left
  strays or halos: weight 100 without kinetic terms has max_cov 71.2 (one stray), and weight 1000 has max_cov 8.05 (4.58
  with kinetic terms). Without seeds, the weights cannot be ranked against each other.
- What remains is within-mode quantization. With ~32 atoms per mode, energy-optimal free atoms reach a radial KS of about .13
  (witness), and the trained preset sits at .15-.20. The gate takes the worst mode and the worst of 5 checks, so a pure-atom
  law trained by this energy objective did not pass. That is a limit of the objective, not of pure atoms: the
  hand-stratified 256-atom witness passes (rate .99, KS .066). Pure-atom placement mechanisms that reach quantile-like
  atoms (for example, quantile-aware atoms) are not ruled out.
- A continuous served law (DV12) reached KS .155 but kept a 3-6σ halo (max_cov 12.2). That is a different served-law
  cohort, and it needed a ka2/e22 base plus prior_lr_mult 32. Both attempts at k3p + DV12 (4 configs) ERRORed before any
  measurement. Base `k3p` failed at Recipe construction, and the `e22` base with `critic_formulation: "k3p"` failed at
  update 0. They are listed off-board on the leaderboard and are not in the tuning budget.

## Holdouts (each run once at a85a34fb1, seed 0, after a predeclared expectation)

The expectations file is `E-confirm/predeclared.md`, sha256 974ab5aa…. Its timing rests only on a local mtime and a locally
written hash, so the predeclaration is self-attested. There is no external or git timestamp.

| holdout | status | result |
|---|---|---|
| `api-ring8-hold` | **ERROR (scientific prerequisite)**; not a hold FAIL | The acquisition prerequisite at update 1200 failed exactly `max_cov_eigen` (3.27) and `max_radial_ks` (.176). hq .963, tv .0156, min_cov .61 and modes 8 held. **The hold phase is unmeasured.** The main expectation held. My refinement "hq probably fails too" was wrong |
| `api-gaussian2d` | **COMPLETE / PASS**, sustained 5/5 (13/24) | Worst of last 5: mean_err .043σ (≤ .10), cov .954..1.000 ([.85, 1.15]), radial KS **.066 (≤ .075, thin margin)**, projection KS .038 (≤ .06). Matches the predicted PASS. The predicted radial KS ≤ .04 was wrong |

The first `api-gaussian2d` launch was stopped at update 42 by a 2-hour tool limit. It left no receipt. Its step-0 and
step-42 failed bounds were seen before the identical, deterministic relaunch, and nothing was changed. It ran at about 8 s per update (7951 s total).

## Controls

- **Restore** (fresh process): rescoring the restored checkpoint is exact on both cases, including after the damage controls.
- **Damage** (same scorer):
  - gaussian1d: every damage flips PASS to FAIL. Generator noise .3 moves mean_err from .013 to .824σ. Prior-row collapse moves std_ratio to .047. Half the rows on one region gives cdf_ks .423.
  - ring8 already fails, so damage there shows metric degradation, not a PASS-to-FAIL flip. Each damage raises the number of failing gates from 2 to 4. The gates do not move monotonically: collapse gives hq 1.0 and max_cov 0 (a single atom), and half-rows lowers max_cov to 1.98.
- **Energy off** (weight 0, same schedule delta, run once):
  - ring8 FAIL. hq .926, tv .191 (locked from update 300), cov .80..96.1, KS .643.
  - gaussian1d PASS 5/5 (14/24) with weaker margins.
  - So gaussian1d does not discriminate the energy mechanism. ring8 and gaussian2d carry the mechanism evidence.
- **Off identity:** at weight 0, 12-update state digests are bit-identical with and without the energy commits for
  k3p particles, k3p MoG, bispf_optimized and atlas. `tests/test_energy_transport.py` keeps this check on develop.
- **Forbidden fallback** (7 receipts): every receipt checks each of the following.
  - The energy fields are present and equal their declared values.
  - continuous_policy is null, serve_average is 0, birth/death is off and output noise is 0.
  - The checkpoint carries `energy_transport` (capacity 4096, population on) and an `energy_generator` stream.
  - The served source is `fast`.
  - The stored samples equal the live `sample(ema=False)` output and differ from EMA.
  - On ring8, the samples equal G(prior atoms[rows]) computed by hand.

## Continuation

Fresh-process resumes from `api_run --steps K` prefixes are exact, with 0 mismatches:
- ring8 K=20: 24 checked steps. The resume crosses the FIFO fill (2560 of 4096 rows filled at K, full at update 32).
- ring8 K=600: 12 checked steps, after the anneal start.
- gaussian1d K=300: 17 checked steps.

The comparison is on observation metrics, not on a full state digest.

Forge checkpoints of population runs are covered by a software test only. On a small k3p population config,
`test_forge_population_checkpoint_round_trip_and_exact_continuation` round-trips the state with its named energy
stream, continues exactly in a fresh context object (same process, not a fresh process) and reports a damaged
checkpoint as incomplete. No Forge run of the selected config has been made.

## Tuning budget

The [leaderboard](leaderboard.md) lists **163 distinct (base, delta) configurations**. A config re-measured on another
source is counted once. Of these:
- 162 are selection configs. 117 come from round 1 and the baseline runs (L1-L4, presets, C-combine, develop). 45 come
  from round 2: 24 pure-atom (E-S1-atoms), 20 continuous-law (E-S2-continuous) and 1 build (E-build).
- 1 is the energy-off control (E-confirm), which was not used for selection.

The board ranks configs within served-law cohorts: 107 pure-atom, 2 DV11, 18 DV12, 35 DV12 with state-selected serving,
and 3 with no readable receipt. These sum to 165 because 2 configs also have a receipt-less re-measurement.

Not counted, and listed off-board on the leaderboard: the 4 intended k3p + DV12 configs from the two nested
`E-S2-continuous/attempt*` directories, which ERRORed before any measurement (see Mechanism). They are non-numerical
outcomes, not FAILs.

Unequal tuning: k3p_energy received that whole selection budget, and no other trainer was retuned against it.

## Integration onto develop

The branch is four cherry-picks onto develop 805713ed5 (rhbi-v7), followed by three more commits: one source fix
(41b2acb2c), one test declaration and this report. The last two change no library source.

- **Conflicts.** The trainer applies develop's rhbi simplex term first, then energy. The streams do not collide: rhbi uses
  seed +6/+7/+8 and energy uses +9. The field lists and test expectations take both sides, and the recipe field count
  moves from 155 to 158 (computed).
- **Task-card-pinned sources stay byte-identical to develop.** `benchmarks/toy_audit/api_contract.py` and
  `particlegan/recipe_compat.py` are pinned by integrated model-toy task cards (12 and 4) under
  `reviewed-model-prs-integration-20261109`, so any byte change needs a new source-binding layer. The fix commit
  therefore moves each energy addition to an unpinned owner:
  - Energy search defaults register in `recipes.py`, which is develop's rhbi pattern.
  - Forge `planning.py` lists the three fields in its projections, as the rhbi PRs did. `planning.py` has no task-card
    or test pin.
  - The caller-objective guard is now a Recipe envelope check. A positive weight outside the GANTrainer envelope is
    refused at construction.
  - The sweep-only gaussian1d tuning-override and Recipe-injection surface is withdrawn. The named `k3p_energy` path does
    not use it, and the sweep receipts keep their original source commits.
- **Pin check.** The task-card pin mismatches equal develop's 18 historical ones.
- **Packet declaration.** The bcap develop-compatibility probe lists the three energy defaults as later inactive Recipe
  defaults in `tests/conftest.py`, as it does for rhbi.
- **Generated views.** The technique inventory and the compiled experiment memory are regenerated with their documented
  commands. The only content change drops the inventory's current-source readout table (see Risks).

A review then added one source fix (859c70e1d) and one report revision:

- **Review fix (859c70e1d).** Forge binds the population `energy_generator` as the named stream
  `("noise", "energy", "population")`, only for recipes that enable population mode, so population checkpoints
  round-trip and continue exactly. Before this fix, `require_consistent_rng` rejected every one of them. The scale
  `s` is no longer clamped to eps: a target set with fewer than two rows or `s <= eps` raises, and the recipe
  requires `batch_size >= 2` and memory 0 or `>= 2`. The unpinned caller-owned hosts refuse a positive weight as
  BLOCKED instead of silently dropping it. No touched file is task-card pinned. The pin mismatches still equal
  develop's 18 cards (26 files).
- **Rerun at 859c70e1d.** `api-ring8-acquire` gives FAIL 0/5 and `api-gaussian1d-acquisition` gives PASS 5/5.
  Against the 41b2acb2c rerun, all 25 observations, the failed bounds, `observations.npz`, the batch sequence, the
  initial state and models and `final-state.pt` are identical (`E-review/rerun-compare.jsonl`, recorded in
  provenance). The fix changes no value the selected run computes.
- **Report revision.** It adds the cohort-ranked leaderboard and its generator, the stray-atom witness, and the
  corrected tuning-budget and mechanism claims.

## Port to master 952c3b62b

The branch `gabehamilton/native-baseline-energy-master` carries this work onto fork master 952c3b62b, which equals
upstream master (Merge pull request #539: rhbi-v9..v12, empf-v8, the `rhbi_optimized` default promotion and 0.10.0).
The develop-based history (through 2c2433df9) is squashed into three commits: the energy capability (93fb228a3,
41b2acb2c, 9f0f87250 and 859c70e1d), the `k3p_energy` preset (16dfd3982) and this report. The withdrawn gaussian1d
tuning-override surface (d447d1cdd, d7ce31e41) is not carried.

- **Conflicts.** Only field lists and their expected counts conflicted. Master's eight new RHBI fields come first and the three
  energy fields follow. The recipe schema moves from 163 to 166 fields (computed with `get_recipe_fields()`), and the
  bcap compatibility probe checks 61 inactive defaults (58 + 3). The energy edits to `training.py`, `recipes.py`,
  `docs/api.md`, `techniques.py` and `tests/conftest.py` are line-identical to the develop branch's. Master's
  simplicial-chart RHBI term still runs before energy in the G phase. `DEFAULT_RECIPE` stays `rhbi_optimized`.
- **Master changes and `k3p_energy`.** The toy hosts, Forge stream binding, prior sampling and the `k3p` preset are
  unchanged on master. Master's trainer, optimizer and autoencoder edits are gated on RHBI or EMPF formulations, the
  `newton_schulz_hybrid_v1` dense profile or AE encoders, and `k3p_energy` uses none of them.
- **Pins.** All 49 task-card-pinned source paths are byte-identical to 952c3b62b. The historical pin drift equals
  master's 18 cards (26 files).
- **Generated views.** `regenerate_technique_inventory.py` and `forge compile --summaries-only` reproduce master's
  committed `technique-inventory.md`, `EXPERIMENT_MEMORY.md` and `compilation.json` byte for byte, and
  `compile --check` is CURRENT. Master's inventory has no current-source readout table, so on this base nothing is
  withheld (compare the "Inventory readouts hidden" risk, which applies to the develop branch).
- **Reproduction: same evidence, re-identified.** `api_run --recipe k3p_energy` at the port's library head (e58b1933a)
  gives `api-ring8-acquire` FAIL 0/5 (0/24) and `api-gaussian1d-acquisition` PASS 5/5 (22/24). Against the 859c70e1d
  review rerun, all 25 observations, the failed bounds, the resolved recipe, `observations.npz`, the batch sequence, the
  initial state and models and `final-state.pt` are identical on both tasks (`E-fork/rerun-compare.jsonl`, recorded in
  provenance). This is not new evidence. The holdout and control receipts (gaussian2d PASS, ring8-hold ERROR, energy
  off, continuation) were not rerun and stay bound to a85a34fb1.
- **Off identity on this base.** With energy at weight 0, 12-update trainer-state digests are identical at
  952c3b62b and at the port head for `k3p` (particles and MoG), `bispf_optimized`, `atlas`, master's default
  `rhbi_optimized` and `empf` (`E-fork/controls/CTRL-off-identity.txt`, same script as the develop control plus the
  two master presets). The four develop-era digests are unchanged.
- **Leaderboard.** The records are unchanged. The committed generator reproduces `leaderboard.md` byte for byte, so it
  was not regenerated.
- **Tests.** 163 targeted files (`tests/test_toy_api*`, `test_recipe*`, `test_forge_*`, `test_training*`, every
  conflicted file, `test_energy_transport.py`, `test_k3p_energy_preset.py` and the energy/kinetic-transport grep hits)
  were run against a 952c3b62b baseline worktree. There are 0 new failures. The 98 failing ids (19 files) are identical
  on master: macOS `/proc/meminfo` (16 files), CUDA-only modules (2) and one native3 publication check. The two
  port-only files pass 27 and 3 tests (`E-fork/tests/SUMMARY.txt`, recorded in provenance).

## Risks

- **Selection bias.** The winner was picked by min-max margin over 162 configs on two tasks, so its margin is optimistic.
- **Single seed.** This is by rule. Single-run variability is unmeasured, so m4096 vs m8192 and w300 vs w1000 cannot be told apart.
- **Served-law cohorts.** The pure-atom result and the DV12 continuous-law results are separate cohorts and must not be combined in one claim. The leaderboard ranks each cohort separately.
- **Global delta.** The schedule fields (lr_anneal_start/floors) are inherited from round 1. Their effect on other tasks is unmeasured.
- **Cost.** The population mode scales as N × M. On gaussian2d (20,000 prior rows) that is a 20,000² cdist per update, ≈ 8 s per update with ≈ 11 GB RSS.
- **Thin gaussian2d margin.** The radial KS margin is .066 against a bound of .075.
- **Self-attested predeclaration.** The predeclaration evidence rests on a local mtime and hash (see Holdouts).
- **Residual host scope.** Caller-owned toy hosts that do not consume the objective now refuse a positive weight as an
  unsupported mechanism (BLOCKED): ring16 BiGAN, the `api_conditionals` MisGAN fixtures, cGAN overlap, CycleGAN,
  Cramer GAN, InfoGAN, PacGAN and the half-frequency vector hosts. Three more hosts run their own loops inside the
  GANTrainer envelope and are pinned byte for byte by task cards: the MisGAN model host (`api_model_misgan.py`), the
  `api_images` joint-word host and the `api_images` conditional image fixtures. Since the PR #1 review fix,
  `Recipe.make_loss()` refuses a positive weight unless the caller declares `binds_energy_transport=True`, which only
  GANTrainer does. These hosts therefore fail at construction, before any update, on every entry path. Forge preflight
  reports BLOCKED. `api_run` records `ERROR`, not `BLOCKED`, because its BLOCKED mapping
  (`api_contract.UnsupportedRecipeOverrides`) and the hosts' own refusal sites are pinned. Recording BLOCKED there
  needs a new source-binding layer for those cards. The fix changes no training numerics.
- **Packet shape.** Forge `public_components` packets (`behavior_adapters.py`, which is pinned) now carry the three energy
  fields at their defaults, as they already do for the rhbi fields. The bcap develop-compatibility probe declares them as
  later inactive defaults (`tests/conftest.py`), the same way it declares the rhbi fields. Forge refuses a positive energy
  weight on that path.
- **Inventory readouts hidden.** Any library source change drops the inventory's current-source Tier 1 readout table
  (`679a26b257b6`, including RHBI 15/16 with 1 fail, plus nine 0/16 not-run rows) until those readouts are refreshed on
  the new source. The inventory itself carries no note about this. It is a generated view that must match its reducer,
  and the publication pins that reducer's sha256, so a hand-written note would fail the sync check. The records are
  unchanged: [`readout-1e8054c27ba98f19d9d9202a.json`](../records/readout-1e8054c27ba98f19d9d9202a.json) holds the
  RHBI readout. The develop->master PR that links the inventory should state that this table is withheld until the
  readouts are re-run on the new source digest.

## Recommendations

1. Do not adopt `k3p_energy` as a default or claim superiority. Keep it as the declared-delta native baseline FAIL for ring8.
2. Stop tuning the pure-atom energy lineage. Its remaining KS gap (.05-.10) is the floor this energy objective reaches
   (energy-optimal free atoms sit at about .13 against the .10 gate), not a tuning gap. It is not a floor of pure atoms:
   hand-stratified atoms pass, so a recipe-owned placement mechanism that reaches quantile-like atoms is still a
   candidate, with its own declared delta and evidence identity.
3. Next mechanism: a recipe-owned continuous served law that works on k3p. Both k3p + DV12 attempts currently ERROR before any measurement (off-board). It needs its own capability, declared delta and evidence identity, measured as a separate served-law cohort.
4. Before population energy runs on large tables: add a declared chunked or subsampled population variant as a new capability. Do not silently change the formula.
5. Spend untouched holdouts (`api-ring8-shift`, ring16) only once a candidate passes ring8. `api-ring8-hold` stays unmeasured until then.
