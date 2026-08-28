"""The `nequip-distill` entry point.

Hydra entry, set up the same way as `nequip.scripts.train.main`: the config
directory defaults to the current working directory, so a config elsewhere is
selected with `-cp <absolute path> -cn <name>`.

One command covers the whole distillation. `run` is a list; `sample` may appear
once, first, and generates a teacher-labeled dataset in `sample_path`. Whatever
follows it -- `train`, `val`, `test` -- is handed to nequip's own trainer, in this
process, with the dataset wired in. Omitting `sample` retrains a student on a
dataset that is already there; omitting the tail builds the dataset and stops.

Student checkpoints land wherever the config's `ModelCheckpoint` puts them, which
in nequip's own configs is this command's hydra output directory. Restarting an
interrupted training is nequip's normal `ckpt_path` restart.
"""

import copy
import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..sample import SPLITS, split_file

# match nequip: full stack traces rather than hydra's truncated ones
os.environ["HYDRA_FULL_ERROR"] = "1"

logger = logging.getLogger(__name__)

# what a `run` list may hold, beyond our own `sample`. Deliberately not the whole
# set nequip accepts -- `predict` is hidden upstream and has no meaning here.
_TRAIN_RUN_TYPES = ("train", "val", "test")
# checked before sampling, so a config that cannot train fails before the teacher
# has been called a few hundred times
_TRAIN_CONFIG_SECTIONS = ("data", "trainer", "training_module")


def _split_run_list(config: DictConfig):
    """Separate our `sample` stage from the stages nequip runs."""
    assert "run" in config, "`run` must be provided in the config"
    is_list = isinstance(config.run, (ListConfig, list))
    runs = list(config.run) if is_list else [config.run]
    if not runs:
        raise ValueError("`run` is empty -- nothing to do")

    do_sample = runs[0] == "sample"
    tail = runs[1:] if do_sample else runs
    if "sample" in tail:
        raise ValueError(
            f"`run: {runs}` -- `sample` can appear at most once, and must come "
            "first: everything after it trains on what it produced"
        )
    for run_type in tail:
        if run_type not in _TRAIN_RUN_TYPES:
            raise ValueError(
                f"`run: {runs}` -- unknown run type {run_type!r}, expected `sample` "
                f"or one of {list(_TRAIN_RUN_TYPES)}"
            )
    return do_sample, tail


def _check_train_config(config: DictConfig) -> None:
    """Everything about a training config we can check before sampling starts."""
    for section in _TRAIN_CONFIG_SECTIONS:
        assert section in config, (
            f"`{section}` must be provided in the config to run "
            f"{list(_TRAIN_RUN_TYPES)} -- this is nequip's own requirement, checked "
            "here so it is not discovered after a full sampling run"
        )
    assert "_target_" in config.data, "`data._target_` must be provided in the config"

    # the dataset is wired in by setting `train_file_path`/`val_file_path`/
    # `test_file_path`, which is `ASEDataModule`'s interface and no other's
    from nequip.data.datamodule import ASEDataModule

    datamodule_cls = hydra.utils.get_class(config.data._target_)
    if not issubclass(datamodule_cls, ASEDataModule):
        raise ValueError(
            f"`data._target_` is {config.data._target_}, but a distillation run reads "
            "its dataset from the three files written by the sampler, which is what "
            "`nequip.data.datamodule.ASEDataModule` takes -- use that one, and leave "
            "its `train_file_path`/`val_file_path`/`test_file_path` unset"
        )
    if config.data.get("split_dataset", None):
        raise ValueError(
            "`data.split_dataset` is set, but sampled datasets come pre-split into "
            "train/val/test files and are never re-split -- re-splitting at train "
            "time is what would let a grown dataset move held-out structures into "
            "training. Remove it."
        )


def _dataset_files(sample_path: Path) -> dict:
    return {split: split_file(sample_path, split) for split in SPLITS}


def _require_dataset(files: dict, sample_path: Path) -> None:
    """A run with no `sample` stage still needs a dataset to train on."""
    missing = [str(p) for p in files.values() if not (p.exists() and p.stat().st_size)]
    if missing:
        raise FileNotFoundError(
            f"`run` has no `sample` stage, so {sample_path} must already hold a "
            f"complete dataset, but these are missing or empty: {missing}"
        )


def _train_config(config: DictConfig, tail: list, files: dict) -> DictConfig:
    """The config nequip's trainer sees: ours, minus our keys, plus the dataset."""
    train_config = copy.deepcopy(config)
    OmegaConf.set_struct(train_config, False)

    train_config.run = list(tail)
    for key in ("sampler", "sample_path"):
        train_config.pop(key, None)

    for split, path in files.items():
        key = f"{split}_file_path"
        existing = train_config.data.get(key, None)
        if existing:
            logger.warning(f"ignoring `data.{key}: {existing}`, overwritten by {path}")
        train_config.data[key] = [str(path)]

    # nequip tests for this key's presence, not its value, so a config that spells
    # out `ckpt_path: null` would send it down the restart path with nothing to load
    if train_config.get("ckpt_path", None) is None:
        train_config.pop("ckpt_path", None)
    return train_config


def _release_teacher(sampler) -> None:
    """Let go of the teacher before the student asks for the same GPU."""
    calculator = getattr(sampler, "calculator", None)
    sampler.calculator = None
    del calculator
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(config: DictConfig) -> None:
    do_sample, tail = _split_run_list(config)
    assert "sample_path" in config, "`sample_path` must be provided in the config"
    if do_sample:
        assert "sampler" in config, "`sampler` must be provided in the config"
    if tail:
        _check_train_config(config)

    sample_path = Path(config.sample_path)
    files = _dataset_files(sample_path)
    n_new = 0

    if do_sample:
        # `sample_path` is a top-level key rather than a sampler argument, because it
        # is also where the trainer is pointed to read the dataset back.
        sampler = instantiate(config.sampler, sample_path=config.sample_path)
        # The sampler stores this in its state file and, on a later run, compares it
        # against the live config to decide whether the dataset on disk is one it
        # would have produced itself. Handed over after instantiation rather than as
        # an argument, because `instantiate` recurses into the arguments it is given
        # looking for things to build, and this dict holds the calculator's own
        # config.
        sampler.sampler_config = OmegaConf.to_container(config.sampler, resolve=True)
        logger.info(f"sampling -> {sample_path}")
        n_total = sampler.generate()
        counts = ", ".join(f"{s}={n}" for s, n in sampler.split_counts.items())
        n_new = n_total - sampler.n_resumed
        logger.info(
            f"{sample_path} holds {n_total} labeled structures ({counts}); "
            f"{n_new} from this run, {sampler.n_resumed} already present"
        )
        if tail:
            _release_teacher(sampler)
    else:
        _require_dataset(files, sample_path)

    if not tail:
        return

    # A restart continues the run the checkpoint came from: it picks up at the
    # restored epoch and stops at the same `max_epochs`. Hand one back together with
    # a dataset that just grew and the new structures get whatever epochs happened
    # to be left over -- none at all if that run reached `max_epochs`. Measured with
    # this guard removed: a checkpoint two epochs into a two-epoch run resumed, ran
    # ONE epoch, and exited 0 (the restored point was the epoch-0 best, see below).
    # Training on an extended dataset means warm-starting from the trained weights
    # with a fresh epoch budget, which is a different thing than a restart and is
    # not built yet, so refuse rather than look like it worked.
    #
    # Note `run_stage` does NOT provide a guard here. nequip advances it after each
    # stage returns, but only Lightning writes checkpoints and only during `fit`, so
    # every checkpoint from a `train`-first run stores `run_stage == 0` however far
    # the run got. A restart always replays train, val and test.
    if n_new and config.get("ckpt_path", None) is not None:
        raise ValueError(
            f"this run added {n_new} structure(s) to {sample_path} and was also "
            f"given `ckpt_path: {config.ckpt_path}`. Restarting from a checkpoint "
            "continues the run that checkpoint came from, which may well have "
            "nothing left to do -- it would not train on the new structures. Drop "
            "`ckpt_path` to train a fresh student on the extended dataset, or drop "
            "`sample` from `run` to resume the interrupted training as it was."
        )

    from nequip.scripts.train import main as nequip_train

    train_config = _train_config(config, tail, files)
    logger.info(f"handing {list(tail)} to nequip, dataset from {sample_path}")
    # Always with the config: called bare, this parses `sys.argv` and creates a
    # second hydra output directory of its own. Passing it keeps everything inside
    # this command's directory, which is also what `${hydra:runtime.output_dir}` in
    # the config resolves to.
    nequip_train(train_config)


if __name__ == "__main__":
    main()
