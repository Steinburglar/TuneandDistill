# TuneandDistill — Agent Context

## Purpose

`nequip` extension pkg. Ships ONE new CLI: **`nequip-distill`**.

Given teacher model artifact loadable as ASE calc + base frame(s):
1. **sample** frames (ASE MD, or rattle/deform of base frames)
2. **label** w/ teacher calc
3. **train** student via nequip's `train()`

Finetuning OUT OF SCOPE (nequip core does it). This repo = distillation only.

**Scope: hackathon one-off, warm-up for possible later extensibility work. NOT a broad
extensible framework.** Reject abstraction/hooks/plugin seams that only pay off hypothetically.
Product pitch = ONE command; any design needing 2 commands or 2 configs is rejected.

Design source: `planning.md` (this repo). Broader framework vision: `../Zero2Tuned/DESIGN.md`.
`../distillation/` is **NOT precedent/source of truth** here (different repo, Snakemake-based).
Cite it ONLY for the rattle algorithm + ASE artifact gotchas below.

## State — 2026-08-28

**Sampling half DONE + verified on GPU (`d61ee43`, RNG fix `96d54c8`). Sampler resume: HALF DONE
(`f9689f3`) — record written + read back, rattle resumes, MD does NOT, every config change
refused.** Student side (train handoff) NOT started.

`nequip_extension_template/sample/`
- `sampler.py` — base `Sampler`. Holds calculator, `base_frames`, `sample_path`, `n_written`,
  `n_resumed`, `split_counts`, `state_interval`, `sampler_config`. Abstract surface = `finished`
  (property) + `step()`. Overridable: `procedure_state()` (default `{}`),
  `restore_progress(blob)` (default RAISES `NotImplementedError` → a sampler not taught to
  resume refuses instead of duplicating everything). Concrete: `split_file(s)` /
  `train_file` / `val_file` / `test_file`, `append(atoms, split)`, `state_file`, `write_state()`,
  `read_state()`, `check_goal(stored)`, `truncate_to(offsets)`, `generate()`.
  Module fns: `frames_digest(frames)`, `flatten(dict)`.
- **resume, as implemented (`f9689f3`)** — `sampler_state.pt` beside the dataset holds
  `{version, sampler_class, goal: {config, base_frames}, progress: {n_written, split_counts,
  offsets, procedure}}`. `offsets` = byte length of each split file. Written every
  `state_interval` structures (default 1) + at end, atomically (`.pt.tmp` + `os.replace`).
  `generate()` order on resume: read → restore counters → `check_goal` → `truncate_to` →
  `restore_progress`. Refuses: unknown `version`, different `sampler_class`, split files with NO
  record, file shorter than recorded offset, `sampler_config is None`, ANY config difference.
  File LONGER than offset → truncated + warned (this fires even at `state_interval=1`, measured).
- `rattle.py` — `RattleSampler`. Owns `label()` (teacher call + `SinglePointCalculator`).
  Resumes: `procedure_state()` = `{"n_steps": ...}`, nothing else. Order is fixed (variant-major)
  and the goal must be unchanged, so the count IS the position.
- `md.py` — `MDSampler`. Langevin NVT, ONE trajectory from `base_frames[0]`, rest ignored (warns).
  CANNOT resume — inherits the base `restore_progress` refusal, no md.py code needed. Needs
  positions + velocities + `rng.bit_generator.state` (Next steps #2).
- `split.py` — free fns, NOT a base-class method. `assign_splits(n, split, seed, policy)`,
  `policy` = `"scattered"` (torch shuffle) or `"blocked"` (contiguous train→val→test ranges).
- `__init__.py` — exports `Sampler`, `RattleSampler`, `MDSampler`.

`nequip_extension_template/scripts/distill.py` — hydra entry, `@hydra.main(version_base=None,
config_path=os.getcwd(), config_name="config")`, same shape as `nequip.scripts.train.main`. Only
`run: [sample]` implemented; anything else raises `NotImplementedError`. Sets
`sampler.sampler_config = OmegaConf.to_container(config.sampler, resolve=True)` AFTER
`instantiate` — NOT as a ctor arg, `instantiate` recurses into args hunting `_target_` and would
build the teacher twice. Logs new-vs-already-present via `sampler.n_resumed`.
`nequip_extension_template/scripts/__init__.py` — added (was missing, needed for import).

`tests/test_resume.py` — 16 tests, ~26 s, CPU only, NO teacher and NO GPU. `pytest` from repo
root (config in `pyproject.toml`). See "Test suite" below.

`pyproject.toml` — `[project.scripts] nequip-distill = "nequip_extension_template.scripts.distill:main"`,
`[project.optional-dependencies] test = ["pytest"]`, `[tool.pytest.ini_options] testpaths`.
Still template-shaped otherwise: `nequip>=0.13.0` too loose (want `>=0.17,<0.18`, not yet
done), `name="TODO"`-style placeholders in `description`/`authors`, `license={file=LICENSE}` but
no LICENSE file. Package name stays `nequip_extension_template` — DELIBERATE, no final name
chosen; rename later touches `pyproject.toml` (`name`, `packages.find.include`, entry-points,
`version.attr`) + every intra-pkg import.

`_keys.py` still registers upstream template placeholder fields (`user_facing_graph_field_name`,
`user_facing_node_field_name`) — unused, delete or replace whenever touched.
`configs/distill_template.yaml` — stale UI spec from before this session's `split_policy`/3-file
design; don't trust its `sampler:` shape, `testartifacts/*.yaml` are the current source of truth.

## Design decisions — sampling half (SETTLED, implemented)

**D1. Sampler owns calculator AND labeling, exactly one calculator, no separate labeler.**
Separate `labeler` slot REJECTED (user). Labeling itself is procedure-specific, not on the base
class: `label()` lives on `RattleSampler` only (static-structure eval); `MDSampler` gets labels
free from the dynamics step (Langevin already evaluates energy/forces every step — MEASURED
0 extra teacher calls, by wrapping `calc.calculate`).

**D2. `step()` does everything for one step**: produce + label + append + advance own state, in
one call. `generate()` just loops `while not self.finished: self.step()`. No separate
produce/label seam (D1). **No guard against a subclass whose `step()` doesn't advance
`finished`** — would loop forever. D2's original "no base-class `checkpoint_every`" is SUPERSEDED
(`f9689f3`): the base class owns the state file, so it owns the cadence — `state_interval`, default
1 (a teacher call costs far more than one `torch.save`).

**D3. Base class owns ONLY three empty boxes**: the three split file paths + `append(atoms,
split)`. It does NOT decide sample count, procedure params, or what goes in them — those are
each subclass's job (no `sample_size` on the base class; rattle's extent =
`len(base_frames) * len(variants)`, MD's = `n_samples`, no shared meaning to force into one key).

**D4. Split assignment is per-procedure-chosen unit, computed once at generation time.**
`assign_splits(n, ...)` labels `n` *items*; each sampler decides what `n` counts. Rattle → n =
base frames (rattles of one base frame are near-duplicates, must stay together). MD → n =
snapshots (assumes `sample_interval` decorrelates — UNVERIFIED, see Open risks).
Apportionment itself = `torch.utils.data.random_split` (nequip uses this too,
`nequip/data/dataset/utils.py:55` — don't reimplement). `split.py` only adds: (a) index→label
inversion (need labels before a dataset object exists, since frames are written straight to 3
files as generated), (b) empty-split → hard `ValueError` where torch only warns.
`split_policy` defaults differ ON PURPOSE: `MDSampler` → `"blocked"` (time-ordered, contiguous
tail holdout stays honest if interval under-decorrelates); `RattleSampler` → `"scattered"` (base
frames have no meaningful order). Both `testartifacts/*.yaml` configs set it explicitly anyway.

**D5. Output = three files** `train.extxyz` / `val.extxyz` / `test.extxyz` in `sample_path`, NOT
one `samples.extxyz`. Splits are frozen at generation, never recomputed at train time — this is
what keeps growing the dataset from silently moving val/test frames into train (the fraction-
based leakage risk that a single-file + `data.split_dataset`-fractions design would have). Once
wired to the trainer (Next steps #2), point `train_file_path`/`val_file_path`/`test_file_path`
at these three files directly; `data.split_dataset` fractions become irrelevant to sampled data.

**D6. Restart precedence for sampling: REVERSED from the original plan.** Live config wins;
sampler diffs stored goal vs. live goal, does not silently ignore the live config. Three-way
distinction: progress state (n_written, offsets, procedure blob) / stored goal / live goal —
stored goal is a diff baseline + legality guard, not behavior-driving.

**PARTLY IMPLEMENTED (`f9689f3`)**: the diff exists (`Sampler.check_goal`) but ANY difference is
FATAL, nothing is a warning yet and nothing is legal yet. Classification (which changes are legal,
which fatal) = Next steps #1.

**D6a. Stored goal = a COPY OF THE `sampler` CONFIG, not a hand-picked list of params (user,
settled).** A hand-written `goal()` per sampler duplicates every ctor kwarg and silently omits any
new one — the exact shape of the `anisotropic_strain_magnitude` bug. `distill.py` hands over
`OmegaConf.to_container(config.sampler, resolve=True)`; the sampler stores it verbatim.
Two things the config alone CANNOT express, so they are stored beside it:
- `base_frames` is a PATH. Edit the file, keep the path, and a config-vs-config diff sees nothing.
  Fixed by `frames_digest(self.base_frames)` (hashes the LOADED structures, not file bytes, so
  re-exporting the same frames in a different text layout is not a false alarm).
- MD uses only `base_frames[0]`, so its digest over all frames is over-strict. Not yet addressed.
NO exclusion list yet (user explicit): `state_interval` and `calculator.device` are compared too,
so a resume on a different device currently REFUSES. Deliberate — exempting keys one at a time is
how a real difference gets waved through. Exclusions land with the classification step.
**Consequence to know:** `state_interval` isn't in the yaml, so it needs `+sampler.state_interval=N`
(hydra ADD not override), and a run started that way must be resumed with the same flag or the
live config is missing the key → `state_interval: 5 -> <not set>`.

**D7. Never pickle sampler/calculator. IMPLEMENTED (`f9689f3`).** Three reasons, in order:
(a) a pickled sampler carries its OLD config, and D6 says the live config wins — so the pickle
buys nothing and makes it easy to miss an attribute; (b) compiled/CUDA-bound torch models are
non-portable across nodes/GPU archs; (c) a pickle is coupled to class layout, so renaming an
attribute silently breaks every existing `sample_path` — a plain dict breaks loudly via the
`version` check. `sampler_state.pt` via `torch.save`, `weights_only=False` on load, contents =
plain numpy/dicts only, THREE byte offsets (one per split file). Truncate-to-offset implemented in
`Sampler.truncate_to`: longer than recorded → truncate + warn; SHORTER, or missing with a nonzero
offset → hard error (record and dataset are not from the same run, nothing to recover).

**D7a. RNG derivation is ASYMMETRIC between samplers (settled, `96d54c8`) — load-bearing for
resume.**
- `RattleSampler`: no streaming RNG. Each structure seeded by `derive_seed(seed,
  frame_key(base_frame), variant_label)`, both module-level fns in `rattle.py`. `frame_key` =
  sha256 of numbers + positions (rounded 8) + cell (rounded 8) + pbc, truncated 16 hex chars.
  Structure depends ONLY on its own identity, never generation order — **nothing to checkpoint
  for rattle's RNG.** Variant identity is a LABEL not a position (`"iso:-0.05"`, `"aniso:0"`) so
  adding a strain magnitude doesn't renumber existing variants. `atoms.info` now carries
  `base_frame`, `base_frame_key`, `variant` (MD still carries `md_step`).
- `MDSampler`: KEEPS one streaming `np.random.default_rng(seed)` — must, trajectory is
  sequential, snapshot n only exists via integrating 1..n-1. **Resuming MD requires
  checkpointing `self.rng.bit_generator.state` alongside positions/velocities** (documented in
  `md.py` docstring, not implemented — see Next steps #2).
- **Open trap: CLOSED (`f9689f3`).** Changing `anisotropic_strain_magnitude` on purpose had the
  same silent effect as the bug the decoupling fixed. Now caught, because D6a stores the whole
  config — as are `seed`, `strain_magnitudes`, `max_displacement_ang`, and base-frame content.
  Tested (`tests/test_resume.py`).

## Design decisions — student side (SETTLED, NOT yet implemented)

**D8. `sample` stays in `run`, earns place by being omissible.**
| `run` | `sample_path` | Meaning |
|---|---|---|
| `[sample, train, val, test]` | fresh | full distillation |
| `[sample]` | fresh | build dataset only |
| `[train, val, test]` | existing, complete | retrain diff student on existing dataset (sweep) |
Assert: `sample` first if present, at most one.

**D9. Call nequip's train in-process, do NOT reimplement.** Mechanism:
`nequip.scripts.train.main(config)` — hydra's `@hydra.main` decorator takes `cfg_passthrough` as
first-class first param (`hydra/main.py` `decorated_main`, verified in installed hydra): no new
hydra folder, no sys.argv parsing, inner fn sees outer's `HydraConfig.get().runtime.output_dir`.
Must ALWAYS pass the config (`train.main()` with no arg parses sys.argv, mints a second hydra
folder). Consequence: `${hydra:runtime.output_dir}` in the config resolves to OUR folder —
template's `ModelCheckpoint.dirpath`/`logger.save_dir` land in the one folder, no rewriting by
us. Version pin still wanted for `run_stage`/restart behavior (`pyproject.toml` →
`>=0.17,<0.18`). Config handed over: strip `sample` from `run`, drop `sampler`/`sample_path`,
point `train_file_path`/`val_file_path`/`test_file_path` at the 3 sample files (supersedes D5's
older single-file plan).

**D10. Progress DERIVED from artifacts, no distill-level program counter.** Sample stage →
sample-side state file (D7). Train/val/test → nequip's own `run_stage` (registered buffer in
checkpoint, `nequip/train/lightning.py:161`), untouched.

**D11. ONE ckpt key `ckpt_path`.** `warm_start_from` proposed + REJECTED (user). Script picks
Path A vs B from whether dataset grew (only thing that knows):
- unchanged → Path A, `ckpt_path` passthrough = crash resume (via `last.ckpt`, D13).
- grew → Path B, rewrite `training_module.model` to `ModelFromCheckpoint` from `best.ckpt`
  (D12), warm start on extended data. Log loudly (expensive, easy accident either way).
- escape hatch: user-written `ModelFromCheckpoint`/`ModelFromPackage` builder respected, untouched.
- **TRAP guarded against:** strip `sample` → nequip sees `run:[train,val,test]` (len 3), a
  completed run stored `run_stage==3`. Bump sample count + keep `ckpt_path` → sampler appends
  frames, nequip loads ckpt, loop never executes, exits 0 having trained on nothing new. Only the
  distill script catches this: record `n_written` before sampling, compare after.

**D12. `best.ckpt` vs `last.ckpt` are different jobs — never say "ckpt_path" generically.**
`last.ckpt` = resume point (Path A). `best.ckpt` = best on monitored metric, warm-start (Path B)
must start from THIS not last-epoch weights (`nequip/scripts/train.py` dispatch loop sets
`ckpt_path="best"` after a `train` stage, confirmed).

**D13. Lightning checkpoint versioning must be OFF for a shared student dir.** Confirmed in
installed lightning `model_checkpoint.py`: version counter (`enable_version_counter=True`
default) makes a second run into the same `dirpath` write `best-v1.ckpt`/`-v2`, etc. **Trap:
unsuffixed `last.ckpt` is then the OLDEST, not newest.** `ModelCheckpoint
(enable_version_counter=False)` gives one always-current `best.ckpt`/`last.ckpt` — PRECONDITION
for automatic training resume via `student_path` below.

**D14. Warm-start best-clobbering → ARCHIVE HOOK, NOT yet designed/implemented.**
`ModelCheckpoint.state_dict()` persists `best_model_score`/`best_model_path` (confirmed,
installed lightning); Lightning restores this on a `ckpt_path` resume but a WARM START has no
`ckpt_path` → best tracking starts from zero → new run can overwrite `best.ckpt` even if worse,
irreversible with versioning off (D13). Plan: hook that MOVES the outgoing `best` checkpoint into
an archive folder before a warm start, keeping main slot for true best on current val set.

**D15. `student_path` — SETTLED, adopt (was open, user resolved this session).** Symmetric to
`sample_path`: stable home for checkpoints + D13 versioning off → training resume becomes
automatic (default = resume from last/best checkpoint in `student_path`; new start = hand a new
`student_path`). Cost accepted: we set `ModelCheckpoint.dirpath` ourselves — first real
intervention in nequip's territory (everything else in D9 passes the trainer config through
untouched). Record next to checkpoint: `student_state.json` w/ `{sample_path, sample_n_frames,
sample_updated, hydra_run_dir, epochs, max_epochs, status, best_metric}`. Convergence readable
via `trainer.early_stopping_callback.stopped_epoch` (0 if never fired), `trainer.current_epoch`,
`trainer.max_epochs`, `trainer.checkpoint_callback.best_model_score` (confirmed present,
installed lightning).

## On-disk layout (settled)

- One hydra timestamped folder per `nequip-distill` command, siblings accumulate. Holds resolved
  config + log (sampling log lines land there too), checkpoints once training begins — nequip's
  own convention (`configs/tutorial.yaml` v0.17.1 sets both `ModelCheckpoint.dirpath` and
  `logger.save_dir` to `${hydra:runtime.output_dir}`).
- `sample_path` lives ELSEWHERE, side by side with the hydra folder, not nested either direction
  — different lifetimes (dataset outlives any command, run folder is per-command). Nesting hydra
  folder INSIDE `sample_path` is REJECTED (tested): hydra creates its run dir at command START,
  so `hydra.run.dir: ${sample_path}/...` makes hydra create `sample_path` itself → sampler sees a
  dir with contents but no state file → hard error every fresh run. Reverse (sample_path as
  subdir of hydra folder) is SAFE, tested fine — just never point `hydra.run.dir` at/inside
  `sample_path`.
- Interruption during SAMPLING: no checkpoint to hand back, `ckpt_path` stays null, resume driven
  entirely by `sample_path`. Interruption during TRAINING: new hydra folder on resume unless
  `student_path` (D15) makes `dirpath` stable — this was the "resume asymmetry", now resolved by
  D15's adoption.
- Hydra does NOT chdir (`hydra.job.chdir` unset → False, confirmed installed 1.3.2) → every
  relative path in a nequip config resolves against LAUNCH CWD, not the hydra run dir. Why
  `${hydra:runtime.output_dir}` is used explicitly rather than relative paths.
- `--multirun` groups siblings under `multirun/<date>/<time>/<job_num>/` instead of `outputs/`.

## OPEN — not yet settled

1. **Provenance.** Single `runs.jsonl` REJECTED (split-brain). Settled instead: each stage's
   record lives with its own artifact — sampling's is `sampler_config.yaml`/state file in
   `sample_path` (once D7 lands); training's is the hydra folder's `.hydra/config.yaml` +
   `student_state.json` (D15) next to the checkpoint.
2. Teacher compiling: plan is DEMAND ready calculator-compatible artifact, no compiling in this
   repo. Confirm w/ user before adding — worth revisiting given the `.pt2` segfault below.
3. Sweep race: parallel sweep runs would all sample into the same 3 files simultaneously. Fix =
   lock file in `sample_path`, undesigned.
4. Two-command workflow (`run:[sample]` then `run:[train,val,test]`) REJECTED as the sweep-race
   fix — defeats one-command product. Still legal per D8, just not the answer.
5. Two separate hydra processes in one command REJECTED — tested, both resolve to the SAME
   output folder when run same second (`${now:...}` identical), timing-dependent failure.

**Scope call:** sampling / student-side are separable in TIME. Sampling built first (held the
only real unknowns: ASE MD state capture, extxyz append+truncate, `step()` shape per sampler).

## nequip 0.17.1 integration facts (verified in installed env)

Env: `/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311` (conda prefix, no activate).

**Teacher → ASE calculator.** `nequip.integrations.ase.NequIPCalculator`
(`nequip/ase/nequip_calculator.py` is a deprecated shim, don't use).
- `from_compiled_model(...)` — `nequip-compile` output only (`.nequip.pth`/`.nequip.pt2`).
- `_from_saved_model(...)` — only path for `.nequip.zip` (nequip-package) or raw `.ckpt`.

**nequip's train entry** (`nequip/scripts/train.py`): required top-level sections `run`, `data`,
`trainer`, `training_module`. `run` accepts ONLY `train`/`val`/`test`/`predict` (or a dict w/
`function` key, UNIMPLEMENTED upstream); at most one `train`. `sample` must be stripped before
delegating.

**Path A vs B precedence** (drives D9/D11):
| Section | On `ckpt_path` restart (Path A) | Guard |
|---|---|---|
| `training_module` | checkpoint wins, live config ignored | warns config ignored |
| `data` | live config, always re-instantiated | none |
| `trainer` | live config, always | none |
| `run` | live config decides stages+order; ckpt's `run_stage` decides only START INDEX | assert live `run` matches ckpt `run` as prefix |
| `global_options` | live, folded into `info_dict` only | none |
Path B = `ModelFromCheckpoint`/`ModelFromPackage` as `model` builder — no conflict guard, only a
version-string warning (`nequip/model/saved_models/checkpoint.py:64-71`).

**Workflow state.** `nequip/scripts/_workflow_utils.py::set_workflow_state(state)` asserts
`state in ["train","package","compile",None]` — cannot register `"distill"`/`"sample"`, private
module, don't try.

**Datamodules.** Base `nequip.data.datamodule.NequIPDataModule`. On-disk ASE files:
`nequip.data.datamodule.ASEDataModule`, kwargs incl. `train_file_path=[]`, `val_file_path=[]`,
`test_file_path=[]`, `split_dataset=[]`, `exclude_keys=[]` (this is what D5/D9's 3-file output
plugs into).

**No prior art in nequip core.** No sampler, rattle, active learning, MD driver.
`nequip/data/_sampler.py::PartialSampler` is an unrelated torch DataLoader sampler. Only MD code
is `nequip/ase/nosehoover.py::NoseHoover` — an NVT thermostat class, not a driver.

**Compiled-artifact findings (this session, matters for OPEN #2):**
- TorchScript compile is DEAD: `nequip-compile --mode torchscript` →
  `ValueError: TorchScript compilation is deprecated and not supported in PyTorch >= 2.10`
  (env has torch `2.11.0+cu128`). AOTInductor is the only mode → compiled artifacts always
  GPU-arch-locked.
- AOTInductor compile needs `module load cuda/12.9.1-fasrc01` (env's pip nvidia headers
  incomplete — `fatal error: crt/host_defines.h`; setting `CPATH` alone is NOT enough).
- **`nequip-compile --target ase` AOTInductor `.pt2` SEGFAULTS on first `get_potential_energy()`**
  call — compiles fine, calculator builds fine, crashes at first forward. Suspect export warning
  `aten._linalg_det.default is missing a c-shim implementation`. UNRESOLVED. Packaged
  `.nequip.zip` on the SAME model works fine — isolated to the `.pt2` path.
- `nequip-compile` from a `.ckpt` needs the checkpoint's training data (rebuilds datamodule →
  `FileNotFoundError`); from a `.nequip.zip` package does not.
- gpu_test node used = A100-SXM4-40GB, compute_cap 8.0.

## Rattle prior art — reused, don't reinvent

`../distillation/scripts/gen_synthetic_geoms.py` algorithm now implemented in `rattle.py`: per
base frame, one structure per entry in `strain_magnitudes` (isotropic volume scaling
`(1+strain)^(1/3)`) + `n_random_strain_samples` random anisotropic strains, each followed by
per-atom rattle bounded by `max_displacement_ang`. Variant-major ordering (confirmed): every base
frame gets variant 0 before any gets variant 1.

**Anisotropic strain bound decoupled from `strain_magnitudes` (`96d54c8`).** `RattleSampler` kwarg
`anisotropic_strain_magnitude: float = 0.05` replaces the derived
`max_strain_magnitude = max(abs(strain_magnitudes))` inherited from `gen_synthetic_geoms.py`. The
derived form coupled two knobs: adding one isotropic scan point silently widened the anisotropic
distribution with no change to any structure's name, so nothing could detect it (measured before
fix: adding a strain magnitude left 20/20 `iso` structures bit-identical but changed 10/10
`aniso`; after fix, all 30 bit-identical). Default 0.05 matches old expression's value for the
default `strain_magnitudes`, so default behaviour unchanged. `testartifacts/rattle_only.yaml`
sets it explicitly.

**Hard-won magnitude lesson**: ±10%/5% strain + 0.5 Å displacement pushed frames OOD for the
teacher, students WORSE than no synthetic data. Halved defaults (±5%/2.5% strain, 0.25 Å
displacement, matching `rattle.py`'s `strain_magnitudes`/`max_displacement_ang` defaults) fixed
it. Rattle displacement must be measured against the STRAINED parent, not the unstrained one
(comparing to unstrained overstates spread) — confirmed correct in `rattle.py`.

## ASE gotchas that bit MDSampler / RattleSampler

- **Snapshot every kept frame w/ `atoms.copy()`.** ASE MD mutates one `Atoms` in place;
  appending the live object gives N copies of the final step.
- **`atoms.copy()` drops `atoms.calc`.** Reattach: `snap.calc = SinglePointCalculator(snap,
  energy=e, forces=f)` (`ase.calculators.singlepoint`). Without it, extxyz gets geometry w/ no
  labels, found out only at student-training time.
- ASE `Langevin(fixcm=True)` (current `md.py` default) does NOT strictly sample the correct NVT
  distribution — deprecated since ASE 3.28, fix is `fixcm=False` + `ase.constraints.FixCom`. Left
  alone because MD is scaffolding; becomes a real correctness issue if MD output is trusted.
- numpy 2: `ndarray.ptp()` is gone, use `np.ptp(arr)`.
- Don't name a scratch script `inspect.py` — shadows stdlib, breaks numpy/ase import.
- **Determinism-test trap:** `frames[0].copy()` has the SAME content hash as `frames[0]` by
  construction (D7a's `frame_key`) — testing "adding a base frame leaves existing structures
  identical" with a copied frame passes trivially, proves nothing. Use a genuinely unused frame
  (source has 200, `base_frames.xyz` uses first 10 — frames 10+ are free for this).

## Artifact gotchas inherited from `../distillation/`

- `nequip-package` output must never be relocated after creation — path baked in (one
  `.nequip.zip` did survive a copy this session, don't rely on that holding in general).
- `ASEDataset` auto-includes `energy`/`forces` regardless of `include_keys`; inconsistent
  per-frame extra ASE properties (`dipole`, `free_energy`) crash batching. Set `exclude_keys`
  broadly — rattle/MD tag DIFFERENT provenance keys (`base_frame` vs `md_step`), must cover both.
- `nequip-compile --mode aotinductor` bakes GPU-arch-specific code — A100-compiled artifact
  invalid on H200.
- `nequip-train -cp <path>` needs ABSOLUTE path. Same applies to `nequip-distill`.

## Sandbox — `testartifacts/`

Tracked configs: `test_distill.yaml` (user's, pre-existing, water/`WaterDataModule`, NOT used by
our configs), `rattle_only.yaml`, `md_only.yaml`. Gitignored: `testartifacts/inputs/`,
`testartifacts/out/`.
- `inputs/teacher.nequip.zip` — copied from
  `../distillation/results/CDP/student_direct/S1b/n200_seed1/model.nequip.zip`. THE teacher,
  loaded eagerly (packaged, not compiled — works, just slower).
- `inputs/teacher.nequip.pt2` — the segfaulting compiled one, kept as evidence only.
- `inputs/base_frames.xyz` — 10 CsH2PO4 frames (64 atoms, periodic), first 10 of
  `../distillation/results/CDP/dft_subset/n200_seed1.xyz` (200-frame REAL DFT subset). DFT
  energy/forces DISCARDED, teacher relabels. 10 chosen so 0.8/0.1/0.1 apportions to 8/1/1.
- Named "base frames" not "seed frames" — collided with `n200_seed1` naming AND with RNG seed.

## Run commands

```bash
CONDA=/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311
export LD_LIBRARY_PATH=$CONDA/lib:${LD_LIBRARY_PATH:-}
# from repo root; hydra does NOT chdir so relative paths resolve to launch cwd; -cp must be ABSOLUTE
$CONDA/bin/python -m nequip_extension_template.scripts.distill -cp $PWD/testartifacts -cn rattle_only
$CONDA/bin/python -m nequip_extension_template.scripts.distill -cp $PWD/testartifacts -cn md_only
```
All runs go through Slurm (user explicit, never login node).

**Last known-good smoke test** (both configs, pre-commit `d61ee43`): `rattle_only` → 50 labeled
structures (train=40, val=5, test=5) into `testartifacts/out/rattle_smoke`. `md_only` → 30
(train=24, val=3, test=3) into `testartifacts/out/md_smoke`. NOTE both predate `f9689f3`, so they
have NO `sampler_state.pt` — pointing a run at them now hard-errors (split files, no record). Same
for any `testartifacts/out/s2_*` left lying around.

**Resume verified on GPU** (A100-SXM4-40GB, `f9689f3`): runs killed at 51/200 and at 80/200 (the
latter at `state_interval=5`, so truncation actually ran) resumed to datasets holding the same 200
structures, no duplicates, each in the same split and the same in-file order as an uninterrupted
run, positions and cells BIT-IDENTICAL.

No pre-commit git hook installed in `.git/hooks/` (config present, hook never run); ruff not in
the nequip311 env (`No module named ruff`) — lint by hand before committing.

## Test suite

`tests/test_resume.py`, 16 tests, ~26 s, run with plain `pytest` from the repo root.

**CPU only, NO teacher, NO GPU — and that is a correctness decision, not just speed (user
explicit: no GPU test).** Calculator = `ase.calculators.lj.LennardJones`, base frames = 12
synthetic 8-atom Ar cells built in a module fixture at three file lengths (10/11/12). Reason: the
real teacher is a compiled model on GPU and two runs over BIT-IDENTICAL geometry disagree by
~3e-6 eV / ~4e-6 eV/Ang (float reductions not associative, order not fixed — MEASURED between two
uninterrupted runs). So with the real teacher, "resumed == uninterrupted" is only checkable to a
tolerance; with LJ it is a comparison of file hashes, which also catches a dropped, duplicated or
mis-filed structure. **So: byte-identity of extxyz is NOT a valid criterion whenever the teacher
runs on GPU.** For manual GPU checks use instead: same structure set, same split per structure,
positions/cell EXACTLY equal, |dE| and |dF| < 1e-5.

Test helpers: `build(...)` mirrors what `distill.py` does — same settings both construct the
sampler AND become `sampler_config`. `kill_after(sampler, n)` wraps `step` to raise after exactly
n appends (a structure count, not a wall-clock timeout, so the kill lands at a known point).

**Suite was mutation-checked**, each mutation caught only by the right tests:
`truncate_to` → no-op ⇒ 2 failures; rattle `restore_progress` forgets `n_steps` ⇒ 3;
`check_goal` early-return ⇒ 3.

## Conventions

- Env: prepend `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` on `GLIBCXX_3.4.31 not found`.
- Real compute → compute node. Smoke-scale sampler tests OK on login.
- `pre-commit` configured: ruff lint+format (line-length 88, double quotes), yamllint,
  whitespace hooks, `fail_fast: true` (see hook-not-installed note above).

## Open risks / unverified

- MD `sample_interval` decorrelation (D4) is an ASSUMPTION, unmeasured — if false, per-snapshot
  scattered splitting leaks; `blocked` default mitigates, nothing detects it.
- MD params in `md.py` are placeholders not recommendations (`friction_per_fs=0.01`, 5-step
  equilibration in earlier smoke run).
- `.pt2` AOTInductor segfault (compiled-artifact findings above) unresolved — blocks OPEN #2
  ("demand a ready compiled artifact") until fixed or worked around.
- Teacher used is a *student* model from `../distillation/`, a stand-in — fine for plumbing, not
  a real teacher. Sanity check available for free: teacher E on an unrattled base frame vs its
  DFT E — −374.61 vs −374.51 eV on frame 0 (64 atoms, ~0.1 eV total).

## Next steps (ordered)

**Work in SMALL steps, plan first, no code until the plan is agreed (user explicit, learned the
hard way this session).** Present a short plain-language plan naming exactly what the step adds
and what it deliberately leaves out; do NOT bundle steps; do NOT write smoke tests or extra
verification that wasn't asked for.

1. **Classify goal changes** (finishes D6). Today ANY config difference is fatal. Split into:
   fatal (`seed`, `max_displacement_ang`, `anisotropic_strain_magnitude`, split params — change
   these and structures on disk were drawn under settings the config no longer describes);
   legal-and-extending (more `strain_magnitudes`, higher `n_random_strain_samples`, more base
   frames, higher `n_samples`); irrelevant (`state_interval`, `calculator.device`). Prefer
   declaring the SMALL legal/irrelevant sets and defaulting everything else to fatal, so a knob
   added later is safe until someone thinks about it. Removing anything already on disk stays
   fatal — a written structure cannot be un-written.
2. **MD trajectory state** — positions + velocities + `rng.bit_generator.state` (D7a), replacing
   the inherited `restore_progress` refusal.
3. **Growing a dataset changes SPLIT ASSIGNMENT, and that is its own problem.** Re-running
   `assign_splits` at a larger total MOVES already-written items between files. Items on disk must
   keep their split; new items must be apportioned to close the gap to the target sizes at the new
   total. Rattle splits per base frame (so the map must be keyed by frame hash, not list index);
   MD splits per snapshot, and with `blocked` each extension gets its OWN contiguous val/test tail
   rather than one tail for the whole trajectory — a real cost, flag it before building.
4. Hand dataset to `nequip.scripts.train.main(config)` per D9 — three `*_file_path` lists, strip
   `sample` from `run`, drop `sampler`/`sample_path`.
5. `student_path` (D15, now settled) + `student_state.json` + D14 archive hook.
