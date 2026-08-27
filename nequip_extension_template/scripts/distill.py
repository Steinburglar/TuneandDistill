"""The `nequip-distill` entry point.

Hydra entry, set up the same way as `nequip.scripts.train.main`: the config
directory defaults to the current working directory, so a config elsewhere is
selected with `-cp <absolute path> -cn <name>`.

At this stage only the `sample` run type is implemented -- the config is read, the
sampler is built from it, and frames are generated and labeled. Handing the
resulting dataset to nequip's trainer comes later.
"""

import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

# match nequip: full stack traces rather than hydra's truncated ones
os.environ["HYDRA_FULL_ERROR"] = "1"

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(config: DictConfig) -> None:
    assert "run" in config, "`run` must be provided in the config"
    is_list = isinstance(config.run, (ListConfig, list))
    runs = list(config.run) if is_list else [config.run]
    if runs != ["sample"]:
        raise NotImplementedError(
            f"`run: {runs}` -- only `run: [sample]` is implemented so far."
        )

    assert "sampler" in config, "`sampler` must be provided in the config"
    assert "sample_path" in config, "`sample_path` must be provided in the config"

    # `sample_path` is a top-level key rather than a sampler argument, because it is
    # also where the trainer will later be pointed to read the dataset back.
    sampler = instantiate(config.sampler, sample_path=config.sample_path)
    logger.info(f"sampling -> {config.sample_path}")
    n_written = sampler.generate()
    counts = ", ".join(f"{s}={n}" for s, n in sampler.split_counts.items())
    logger.info(
        f"wrote {n_written} labeled structures to {config.sample_path} ({counts})"
    )


if __name__ == "__main__":
    main()
