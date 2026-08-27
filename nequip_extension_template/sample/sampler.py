"""Base sampler.

Deliberately minimal. All any sampler shares is the teacher calculator, the frames
it starts from, and where the dataset is written. Everything else -- how many
structures there are, what a step consists of, when it is finished, and which split
a structure belongs to -- belongs to the procedure, so it belongs to the subclass.

The base class knows only that the dataset is split three ways and where each part
is written. It does not decide what goes where: see :mod:`.split`.

:meth:`Sampler.step` does the whole job for one step: produce the next structure or
structures, label them, append them to the dataset, and advance the procedure's own
state. Producing and labeling are not separated, because for MD-type procedures they
are not separate events -- the dynamics needs the energy and forces to take the step,
so the label is already in hand. Splitting them would either pay the teacher twice or
require a cache across the seam.
"""

from pathlib import Path
from typing import Union

from ase import Atoms
from ase.io import read, write

from .split import SPLITS


class Sampler:
    """Generate teacher-labeled structures into ``sample_path``.

    Parameters
    ----------
    calculator
        The teacher, as an ASE calculator.
    base_frames
        Path to a file ASE can read, holding the structures the procedure starts
        from. These may carry labels of their own (the CDP frames carry DFT energies
        and forces); those labels are not used and do not reach the output.
    sample_path
        Output directory. Passed in by the ``nequip-distill`` script from the
        top-level ``sample_path`` config key, not from the ``sampler`` section.
    """

    def __init__(
        self,
        calculator,
        base_frames: Union[str, Path],
        sample_path: Union[str, Path],
    ):
        self.calculator = calculator
        self.base_frames = read(str(base_frames), index=":")
        self.sample_path = Path(sample_path)
        self.n_written = 0
        self.split_counts = {s: 0 for s in SPLITS}

    # ------------------------------------------------------------------ subclass API

    @property
    def finished(self) -> bool:
        """Whether the procedure has nothing left to do. Defined by the subclass.

        There is no `sample_size` on the base class on purpose: a rattle procedure is
        naturally bounded by its base frames and variants, an MD procedure by a step
        count and a sampling interval. Forcing both through one number would misstate
        at least one of them.
        """
        raise NotImplementedError

    def step(self) -> None:
        """Produce, label, and append the next structure(s), and advance state.

        Implemented by subclasses, including how the teacher labels are obtained --
        a static procedure calls the calculator on each structure, MD gets energy and
        forces from the dynamics it is already running -- and which split each
        structure belongs to. Use :meth:`append` to write to the dataset.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- dataset files

    def split_file(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {list(SPLITS)}")
        return self.sample_path / f"{split}.extxyz"

    @property
    def train_file(self) -> Path:
        return self.split_file("train")

    @property
    def val_file(self) -> Path:
        return self.split_file("val")

    @property
    def test_file(self) -> Path:
        return self.split_file("test")

    def append(self, atoms: Atoms, split: str) -> None:
        """Append one labeled structure to the named split."""
        path = self.split_file(split)
        with open(path, "a") as f:
            write(f, atoms, format="extxyz")
        self.n_written += 1
        self.split_counts[split] += 1

    # --------------------------------------------------------------------- main loop

    def generate(self) -> int:
        """Step until finished. Returns how many structures were written."""
        self.sample_path.mkdir(parents=True, exist_ok=True)
        existing = [
            str(self.split_file(s)) for s in SPLITS if self.split_file(s).exists()
        ]
        if existing:
            raise FileExistsError(
                f"{existing} already exist. This sampler cannot resume yet -- delete "
                "them, or point `sample_path` somewhere else."
            )
        while not self.finished:
            self.step()
        return self.n_written
