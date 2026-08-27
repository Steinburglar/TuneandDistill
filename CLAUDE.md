# TuneandDistill — Agent Context

## Purpose

`nequip` extension pkg. Ships ONE new CLI: **`nequip-distill`**.

Given teacher model artifact loadable as ASE calc + seed frame(s):
1. **sample** frames (ASE MD, or rattle/deform of seeds)
2. **label** w/ teacher calc
3. **train** student via nequip's `train()`

Finetuning OUT OF SCOPE (nequip core does it). This repo = distillation only.

**Scope: hackathon one-off, warm-up for possible later extensibility work. NOT a broad
extensible framework.** Reject abstraction/hooks/plugin seams that only pay off hypothetically.
Simplest thing that makes the two samplers work. Product pitch = ONE command; any design
needing 2 commands or 2 configs is rejected.

Design source: `planning.md` (this repo). Broader framework vision: `../Zero2Tuned/DESIGN.md`.
`../distillation/` is **NOT precedent/source of truth** here (user explicit — different repo,
Snakemake-based, different situation). Cite it ONLY for the rattle algorithm + ASE artifact
gotchas below. Its dir layout (two-tree, `results/<...>/`) is its own deviation, don't generalize.

## State — 2026-08-27

Design settled (below). **No code written yet.**

- `nequip_extension_template/sample/sampler.py` — abstract `Sampler` stub, predates design,
  CONFLICTS w/ D5 (`from_sampler_checkpoint` classmethod is wrong shape). Not importable (no
  `sample/__init__.py`).
- `nequip_extension_template/scripts/distill.py` — EMPTY. Home of `nequip-distill` hydra entry.
- `MDSampler`, `RattleSampler` — neither written.
- No labeling code, no `project.scripts` entry, no tests.
- Package name stays `nequip_extension_template` — DELIBERATE, no final name chosen. Rename
  later touches `pyproject.toml` (`name`, `packages.find.include`, entry-points, `version.attr`)
  + every intra-pkg import.
- `_keys.py` still registers upstream template placeholder fields (`user_facing_graph_field_name`,
  `user_facing_node_field_name`). Delete or replace.
- `configs/distill_template.yaml` — COMPLETE UI spec: nequip v0.17.1 `configs/tutorial.yaml`
  verbatim + distill additions (`run` w/ `sample`, `sample_path`/`ckpt_path`, `sampler:` section).
  Parses; all 4 nequip-required sections present. `_target_`s are FORWARD REFS to code not yet
  written (`nequip_extension_template.sample.MDSampler`). Upstream copy kept at
  `<scratchpad>/tutorial.yaml` for diffing.
- `pyproject.toml` still template-shaped: `nequip>=0.13.0` too loose, want `>=0.17,<0.18` (not
  yet done). `name="TODO"`-style placeholders in `description`/`authors`, `license={file=LICENSE}`
  but no LICENSE file present. Don't "fix" the package name (deliberate, see above).

## Design decisions — SETTLED

**D1. Sampler owns calculator AND labeling, exactly one calculator, no separate labeler.**
MD evals energy/forces every step so labels are free; splitting sampling from labeling pays
teacher twice or needs a cache across the seam. Separate `labeler` slot REJECTED (user).

**D2. Nested generate.** Inner `step()` advances procedure state, returns one labeled frame;
outer `generate(n)` loops, appends to disk, checkpoints every `checkpoint_every` frames
(frame-count interval, not wall-clock, user-overridable base-class kwarg).

**D3. `sample_path` — single top-level key, dual role: output destination AND resume pointer.**
Asymmetric vs `ckpt_path` on purpose (`ckpt_path` = pure backward reference).
- Discriminator = presence of `sampler_state.pt`, NOT dir emptiness. Absent → fresh. Present →
  resume. Dir w/ files but NO state file → hard error (guards typo'd paths).
- MUST be explicit, no default, never timestamped (timestamped default can never be resumed).
```
<sample_path>/
├── samples.extxyz       # append-only, generated + labeled frames
├── sampler_state.pt     # overwritten; procedure params + progress
└── sampler_config.yaml  # resolved sampler config, provenance only
```

**D4. Restart precedence mirrors nequip's own split:**
- procedure-defining (sampler class, temperature, timestep, rattle magnitudes, teacher artifact
  path) → **state file wins, live config IGNORED, warn.** (`training_module` analog.)
- budget/plumbing (`sample_size`, `checkpoint_every`, `sample_path`, device) → **live config
  wins.** (`trainer`/`run` analog.)
- Guard: assert stored sampler class == instantiated class.

**D5. Never pickle sampler/calculator.** Compiled/CUDA-bound torch models non-portable across
nodes/GPU archs. `sampler_state.pt` via `torch.save`, `weights_only=False` on load. Contents:
`sampler_class` (guard), `n_written`, byte offset into `samples.extxyz`, subclass `state` dict
(plain numpy/dicts only).
- **Truncate rule (makes restart correct):** append-then-save-state isn't atomic → on resume
  truncate `samples.extxyz` to recorded byte offset. Byte offset (not frame count) keeps
  truncation O(1).
- Subclass API: `state_dict()`/`load_state_dict(d)`. Instance `load()` after hydra instantiate
  (NOT a classmethod — stub's `from_sampler_checkpoint` is the wrong shape, would have to
  rebuild the calculator itself).

**D6. `sample` stays in `run`, earns place by being omissible.**
| `run` | `sample_path` | Meaning |
|---|---|---|
| `[sample, train, val, test]` | fresh | full distillation |
| `[sample]` | fresh | build dataset only |
| `[train, val, test]` | existing, complete | retrain diff student on existing dataset (sweep) |
Assert: `sample` first if present, at most one.

**D7. Call nequip's train in-process, do NOT reimplement.** Reimplementing re-owns restart +
lightning + loggers + `run_stage`, rots on next nequip release.
- **Mechanism: `nequip.scripts.train.main(config)`** — NOT `__wrapped__` (superseded, was wrong
  to rely on). Hydra's `@hydra.main` decorator takes `cfg_passthrough` as first-class first
  param (`hydra/main.py` `decorated_main`, verified in installed hydra): `if cfg_passthrough is
  not None: return task_function(cfg_passthrough)`. No hydra init, no new run folder, no
  sys.argv parsing. TESTED empirically: 0 new folders created either way, inner fn sees outer's
  `HydraConfig.get().runtime.output_dir`.
- RULE: must ALWAYS pass the config. `train.main()` w/ no arg parses sys.argv, mints a second
  hydra folder.
- Consequence: `${hydra:runtime.output_dir}` in the config resolves to OUR folder — template's
  `ModelCheckpoint.dirpath`/`logger.save_dir` land in the one folder, no config rewriting by us.
- Version pin still wanted for `run_stage`/restart behavior (`pyproject.toml` → `>=0.17,<0.18`),
  not because of `__wrapped__`.
- Config handed over: strip `sample` from `run`, drop `sampler`/`sample_path`, inject
  `data.split_dataset.file_path = <sample_path>/samples.extxyz`. Nothing else modified.

**D8. Progress DERIVED from artifacts. No distill-level program counter.**
sample stage → `sampler_state.pt` (absent=start, `n_written<sample_size`=resume, else skip).
train/val/test → nequip's own `run_stage` (registered buffer in checkpoint,
`nequip/train/lightning.py:161`), untouched.

**D9. ONE ckpt key `ckpt_path`.** `warm_start_from` proposed + REJECTED (user): 2 keys always
same value, "warmness" covers data+model together. Script picks Path A vs B from whether
dataset grew (only thing that knows):
- unchanged → Path A, `ckpt_path` passthrough = crash resume
- grew → Path B, rewrite `training_module.model` to `ModelFromCheckpoint` = warm start
- escape hatch: user-written `ModelFromCheckpoint`/`ModelFromPackage` builder respected, left alone.
- **TRAP guarded against:** strip `sample` → nequip sees `run:[train,val,test]` (len 3), a
  completed run stored `run_stage==3`. Bump `sample_size` + keep `ckpt_path` → sampler appends
  frames, nequip loads ckpt, loop never executes, exits 0 having trained on nothing new. Only
  the distill script catches this: record `n_written` before sampling, compare after.

**D10. Implicit state-dependence ACCEPTED**, conditional on: `--dry-run` printing the plan
before acting, + provenance on disk so an artifact can never lie about its own history.

**D11. `best.ckpt` vs `last.ckpt` are different jobs — never say "ckpt_path" generically.**
- `last.ckpt` = resume point (optimizer state + epoch ctr) → crash-resume (Path A) uses this.
- `best.ckpt` = best on monitored metric → warm-start (D9 Path B) must start from THIS, not
  last-epoch weights. nequip's own dispatch loop sets `ckpt_path="best"` after a `train` stage
  so following val/test use best (`nequip/scripts/train.py` dispatch loop, confirmed).

**D12. Warm-start best-clobbering → ARCHIVE HOOK (user's call, NOT yet designed/implemented).**
- Problem: `ModelCheckpoint.state_dict()` persists `best_model_score`/`best_model_path`
  (confirmed in installed lightning). Lightning restores callback state on a `ckpt_path` resume
  (crash-resume correctly remembers prior best). A WARM START has no `ckpt_path` → callback
  state NOT restored → best tracking starts from zero → new run overwrites `best.ckpt` even if
  worse. Irreversible w/ versioning off (D13).
- Plan: hook that MOVES the outgoing `best` checkpoint into an archive folder ("best as of the
  sample set at this time"), keeping main slot for true best on current val set. Deferred w/ rest
  of student side.

**D13. Lightning checkpoint versioning must be OFF for a shared student dir.**
Confirmed in installed lightning `model_checkpoint.py`: `_get_metric_interpolated_filepath_name`
and `_save_last_checkpoint` both run a version counter when `enable_version_counter=True`
(default). Within one run, files overwrite cleanly; a NEW run into the same `dirpath` sees the
existing file and writes `best-v1.ckpt`/`last-v1.ckpt`, then `-v2`, etc.
- **Trap: unsuffixed `last.ckpt` is the OLDEST, not newest.** Pointing `ckpt_path` at
  `<dir>/last.ckpt` after several runs silently loads run 1's weights.
- `ModelCheckpoint(enable_version_counter=False)` gives one always-current `best.ckpt`/
  `last.ckpt` — PRECONDITION for automatic training resume (see Resume asymmetry below). Only
  applies once `dirpath` is explicit; nequip's `configs/tutorial.yaml` already sets
  `dirpath`/`save_dir` explicitly to `${hydra:runtime.output_dir}`, so no other fallback path
  (version_N-named dirs, `default_root_dir`) is ever exercised by this template.

### Behavior matrix
| Dataset after sampling | Student init | Behavior |
|---|---|---|
| unchanged | `ckpt_path` set | Path A crash-resume (via `last.ckpt`). Already complete → no-op. |
| unchanged | neither | fresh train on existing dataset (student sweep). |
| **grew** | `ckpt_path` set | **Path B** — `ModelFromCheckpoint` from `best.ckpt`, warm start on extended data. Log loudly. |
| grew | neither | full retrain from scratch on bigger dataset. Legitimate, log loudly (expensive, easy accident). |

## On-disk layout (settled)

- One hydra timestamped folder per `nequip-distill` command, siblings accumulate. Holds resolved
  config + log from the start (sampling log lines land there too), checkpoints once training
  begins. This is nequip's own convention: its `configs/tutorial.yaml` (v0.17.1) sets BOTH
  `ModelCheckpoint.dirpath` and `logger.save_dir` to `${hydra:runtime.output_dir}` → ONE tree.
  `../distillation/`'s two-tree layout is its own deviation, not nequip's convention.
- `sample_path` lives ELSEWHERE, side by side, not nested either direction. Different
  lifetimes — dataset outlives any command, run folder is per-command.
  - **Nesting hydra folder inside `sample_path` is REJECTED** (tested, was own proposal): hydra
    creates its run dir at command START before our code runs, so `hydra.run.dir:
    ${sample_path}/...` makes hydra create `sample_path` itself → sampler sees dir w/ contents
    but no `sampler_state.pt` = D3 hard error, every fresh run.
  - Reverse (sample_path as subdir of hydra folder) is SAFE — hydra only creates its run dir +
    `.hydra/`, so `<hydra_dir>/samples` won't pre-exist. TESTED:
    `hydra.run.dir: ${sample_path}/students/${now:...}`-style interpolation of a top-level user
    config key works fine; just don't point `hydra.run.dir` at/inside `sample_path`. No guard
    designed yet for this misuse.
- Interruption during SAMPLING: no checkpoint to hand back, `ckpt_path` stays null, resume
  driven entirely by `sample_path`. Dead hydra folder is just a log.
- Interruption during TRAINING: new hydra folder on resume, old one never written again, model
  history is a chain across siblings linked only by the typed `ckpt_path`. This is the resume
  asymmetry — see below.
- Per-epoch metrics per-run: WandbLogger → wandb server; CSVLogger → `metrics.csv` in run
  folder. NOT in `sample_path`/any student path.
- `--multirun` groups siblings under `multirun/<date>/<time>/<job_num>/` instead of `outputs/`.
- **Hydra does NOT chdir** (`hydra.job.chdir` unset → False for hydra ≥1.2, confirmed installed
  1.3.2) → every relative path in a nequip config resolves against LAUNCH CWD, never the hydra
  run dir. Why `${hydra:runtime.output_dir}` is used explicitly rather than relative paths.

## Resume asymmetry — live open question (NOT resolved)

- Sampling resume = automatic (stable `sample_path` + state file, re-run same command).
- Training resume = manual (hand-edit checkpoint path into config), AND a second interruption
  needs a DIFFERENT path since each run makes a new hydra folder. Inherited from `nequip-train`,
  worse for us since we promise one command. Root cause: sampling has a stable home, training
  does not.
- Leaning (b), NOT decided: add `student_path` symmetric to `sample_path` (stable home for
  checkpoints + D13 versioning off) → training resume becomes automatic. COST: we'd have to set
  `ModelCheckpoint.dirpath` ourselves, breaking "pass trainer config through untouched" — first
  real intervention in nequip's territory.
- Option (d) rejected as DEFAULT: each hydra run owns its own sample set. Kills resume + student
  sweep, re-pays teacher labeling cost every run. Already expressible by user in one line
  (`sample_path: ${hydra:runtime.output_dir}/samples`) so needs no code/key — we choose a
  DEFAULT (stable), not a model.
- If `student_path` adopted: record next to checkpoint, e.g. `student_state.json` w/
  `{sample_path, sample_n_frames, sample_updated, hydra_run_dir, epochs, max_epochs, status,
  best_metric}`. Convergence readable, not guessed: `trainer.early_stopping_callback.stopped_epoch`
  (0 if never fired), `trainer.current_epoch`, `trainer.max_epochs`,
  `trainer.checkpoint_callback.best_model_score` (all confirmed present, installed lightning).

## OPEN — decided NOT yet settled

1. **Split leakage.** `data.split_dataset` uses FRACTIONS — growing `sample_size` moves split
   boundaries, frames that were val become train, warm-started student already saw them.
   Silently flatters val/test on exactly the D9 Path-B route; D9's frictionless warm start
   raises the stakes. Options: fixed seed + accept, or freeze assignment in `sample_path`
   (`split.json`, frame idx → split, extended as frames appended). Leaning freeze, ~15 lines.
2. **Provenance.** `runs.jsonl` single file in `sample_path` REJECTED (split-brain — record
   would live apart from the run it describes). Settled instead: **each stage's record lives
   with its own artifact** — sampling's record is `sampler_config.yaml`/`sampler_state.pt` in
   `sample_path` (D3, already exists); training's record is the hydra folder's `.hydra/
   config.yaml` + (if `student_path` adopted) `student_state.json` next to the checkpoint. Free
   given the resume work — stable paths exist for resume anyway.
3. Teacher compiling: plan is DEMAND ready calculator-compatible artifact, no compiling in this
   repo. Confirm w/ user before adding.
4. `samples.extxyz` single file vs sharded (`samples_00000.extxyz`, …).
5. Whether sampler output wants its own datamodule (no train/val/test split needed) or just
   writes files.
6. Sweep race: parallel sweep runs all sample into one file simultaneously. Fix = lock file in
   `sample_path`, undesigned.
7. Two-command workflow (`run:[sample]` then `run:[train,val,test]`) REJECTED as the sweep-race
   fix — defeats one-command product. Still legal per D6, just not the answer.
8. Two separate hydra processes in one command REJECTED — tested, both resolve to the SAME
   output folder when run same second (`${now:...}` identical), timing-dependent failure,
   unnecessary given item 2's per-artifact provenance.

**Scope call:** the two halves (sampling / student-side) are separable in TIME — nothing about
sampling depends on any unresolved student-side question. Build sampling half first (holds the
only real unknowns: ASE MD state capture, extxyz append+truncate, `step()` shape per sampler).
MVP smaller than full design: zero provenance + hand-typed `ckpt_path` (nequip's own behavior)
already useful; sampling auto-resume is the genuinely new+needed part.

## nequip 0.17.1 integration facts (verified in installed env)

Env: `/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311` (conda prefix, no activate).

**Teacher → ASE calculator.** `nequip.integrations.ase.NequIPCalculator`
(`nequip/ase/nequip_calculator.py` is a deprecated shim, don't use).
- `from_compiled_model(compile_path, device="cpu", chemical_species_to_atom_type_map=None,
  neighborlist_backend=..., **kw)` — `nequip-compile` output only (`.nequip.pth`/`.nequip.pt2`).
- `_from_saved_model(model_path, device="cpu", ..., compile_mode=..., **kw)` — only path for
  `.nequip.zip` (nequip-package) or raw `.ckpt`.

**nequip's train entry** (`nequip/scripts/train.py`): required top-level sections `run`, `data`,
`trainer`, `training_module`. `run` validation accepts ONLY `train`/`val`/`test`/`predict`, or a
dict w/ `function` key (UNIMPLEMENTED upstream); at most one `train`. `sample` must be stripped
before delegating.

**Path A vs B precedence** (drives D4/D9):
| Section | On `ckpt_path` restart (Path A) | Guard |
|---|---|---|
| `training_module` | checkpoint wins, live config ignored | warns config ignored |
| `data` | live config, always re-instantiated | none |
| `trainer` | live config, always | none |
| `run` | live config decides stages+order; ckpt's `run_stage` decides only START INDEX | assert live `run` matches ckpt `run` as a prefix |
| `global_options` | live, folded into `info_dict` only | none |
Path B = `ModelFromCheckpoint`/`ModelFromPackage` as `model` builder — live config controls
everything, builder only supplies `model`. No conflict guard, only version-string warning
(`nequip/model/saved_models/checkpoint.py:64-71`).

**Workflow state.** `nequip/scripts/_workflow_utils.py::set_workflow_state(state)` asserts
`state in ["train","package","compile",None]`. Cannot register a `"distill"`/`"sample"` state —
private module, don't try.

**Datamodules.** Base `nequip.data.datamodule.NequIPDataModule`. On-disk ASE files:
`nequip.data.datamodule.ASEDataModule` (kwargs `seed, train_file_path=[], val_file_path=[],
test_file_path=[], predict_file_path=[], split_dataset=[], transforms=[], ase_args={},
include_keys=[], exclude_keys=[], key_mapping={}`).

**Extension mechanism.** `nequip/__init__.py` iterates `entry_points(group="nequip.extension")`,
`.load()`s only those literally named `init_always`. Already declared in `pyproject.toml`.
`nequip-distill` itself is a plain `project.scripts` entry, NOT this group.

**No prior art in nequip core.** No sampler, rattle, active learning, MD driver.
`nequip/data/_sampler.py::PartialSampler` is a torch DataLoader sampler, unrelated. Only MD code
is `nequip/ase/nosehoover.py::NoseHoover` — an NVT thermostat class, not a driver.

## Rattle prior art — reuse, don't reinvent

`../distillation/scripts/gen_synthetic_geoms.py`: per seed frame, one structure per entry in
`strain_magnitudes` (isotropic volume scaling `(1+strain)^(1/3)`) + `n_random_strain_samples`
random symmetric-anisotropic strains, each followed by per-atom rattle bounded by
`max_displacement_ang`. Yield per seed frame = `len(strain_magnitudes) + n_random_strain_samples`.

**Hard-won magnitude lesson**: ±10%/5% strain + 0.5 Å displacement pushed frames OOD for the
teacher, students WORSE than no synthetic data. Halving to ±5%/2.5% + 0.25 Å fixed it. Default
conservative.

## ASE gotchas that will bite MDSampler

- **Snapshot every kept frame w/ `atoms.copy()`.** ASE MD mutates one `Atoms` in place;
  appending the live object gives N copies of the final step.
- **`atoms.copy()` drops `atoms.calc`, i.e. drops labels.** Reattach:
  `snap.calc = SinglePointCalculator(snap, energy=e, forces=f)`
  (`ase.calculators.singlepoint`). Without it, extxyz gets geometry w/ no energy/forces, found
  out only at student-training time.

## Artifact gotchas inherited from `../distillation/`

- `nequip-package` output must never be relocated after creation — path baked in.
- `ASEDataset` auto-includes `energy`/`forces` regardless of `include_keys`; inconsistent
  per-frame extra ASE properties (`dipole`, `free_energy`) crash batching. Set `exclude_keys`
  broadly on generated data.
- `nequip-compile --mode aotinductor` bakes GPU-arch-specific code — A100-compiled artifact
  invalid on H200.
- `nequip-train -cp <path>` needs ABSOLUTE path. Same will apply to `nequip-distill`.

## Conventions

- Env: prepend `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` on `GLIBCXX_3.4.31 not found`.
- Real compute → compute node. Smoke-scale sampler tests OK on login.
- `pre-commit` configured: ruff lint+format (line-length 88, double quotes), yamllint,
  whitespace hooks. `fail_fast: true`.

## Next steps (ordered)

1. Write base `Sampler` per D2/D4/D5, replacing conflicting stub in
   `nequip_extension_template/sample/sampler.py`. Add `sample/__init__.py`.
2. Write `RattleSampler` (port algorithm above).
3. Write `MDSampler` (no MD driver in nequip; only `nequip/ase/nosehoover.py::NoseHoover`
   thermostat class).
4. Only after sampling runs: wire training via plain `ckpt_path` passthrough, then revisit
   `student_path` (Resume asymmetry option b) + record + D12 archive hook.
