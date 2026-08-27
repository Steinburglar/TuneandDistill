"""Rattle/deform sampler.

Ported from ``../distillation/scripts/gen_synthetic_geoms.py``. Each base frame is
deformed and then rattled, once per *variant*:

* one isotropic volume scan point per entry in ``strain_magnitudes`` -- the cell is
  scaled by ``(1 + strain) ** (1/3)`` with the atoms scaled along with it
* ``n_random_strain_samples`` structures with a random symmetric anisotropic strain,
  bounded by the largest ``strain_magnitudes`` entry

followed in both cases by a per-atom random displacement of up to
``max_displacement_ang``.

Keep the magnitudes conservative. In the reference work, +/-10%/5% strain with 0.5 Ang
displacement pushed structures outside the teacher's domain and produced students
*worse* than using no synthetic data at all; halving to +/-5%/2.5% with 0.25 Ang
fixed it.
"""

from typing import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from .sampler import Sampler
from .split import assign_splits


def isotropic_strain_matrix(volumetric_strain: float) -> np.ndarray:
    return np.eye(3) * (1.0 + volumetric_strain) ** (1.0 / 3.0)


def random_anisotropic_strain_matrix(rng, max_magnitude: float) -> np.ndarray:
    strain = rng.uniform(-max_magnitude, max_magnitude, size=(3, 3))
    return np.eye(3) + 0.5 * (strain + strain.T)


def rattle_positions(rng, atoms: Atoms, max_displacement_ang: float) -> Atoms:
    directions = rng.normal(size=(len(atoms), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(0.0, max_displacement_ang, size=(len(atoms), 1))
    atoms.positions += directions * magnitudes
    return atoms


class RattleSampler(Sampler):
    """Deform and rattle each base frame, once per variant.

    Parameters
    ----------
    strain_magnitudes
        Volumetric strains for the isotropic scan, one variant each.
    n_random_strain_samples
        Number of random anisotropic-strain variants per base frame.
    max_displacement_ang
        Upper bound on the per-atom displacement, in Angstrom.
    seed
        RNG seed for the strains and displacements.
    split
        Target train/val/test fractions.
    split_seed
        Seed for the shuffle that decides which base frame lands in which split.
    split_policy
        ``"scattered"`` (default) or ``"blocked"``. Base frames have no meaningful
        order here, so scattered is the sensible default; ``"blocked"`` hands out
        contiguous ranges of the input file instead.

    The split is assigned per base frame: the structures rattled from one base frame
    are near-duplicates of each other, so splitting them apart would put effectively
    the same structure in both train and validation.
    """

    def __init__(
        self,
        strain_magnitudes: Sequence[float] = (-0.05, -0.025, 0.0, 0.025, 0.05),
        n_random_strain_samples: int = 3,
        max_displacement_ang: float = 0.25,
        seed: int = 0,
        split: dict = None,
        split_seed: int = 0,
        split_policy: str = "scattered",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strain_magnitudes = [float(s) for s in strain_magnitudes]
        self.n_random_strain_samples = int(n_random_strain_samples)
        self.max_displacement_ang = float(max_displacement_ang)
        self.max_strain_magnitude = (
            max(abs(s) for s in self.strain_magnitudes)
            if self.strain_magnitudes
            else 0.0
        )
        self.rng = np.random.default_rng(seed)
        self.n_steps = 0

        self.variants = [("iso", s) for s in self.strain_magnitudes]
        self.variants += [("aniso", i) for i in range(self.n_random_strain_samples)]
        if not self.variants:
            raise ValueError(
                "this procedure has no variants -- `strain_magnitudes` is empty and "
                "`n_random_strain_samples` is 0"
            )

        self.base_frame_split = assign_splits(
            len(self.base_frames), split, split_seed, split_policy
        )

    @property
    def n_total(self) -> int:
        """One structure per (base frame, variant) pair. This is the whole procedure."""
        return len(self.base_frames) * len(self.variants)

    @property
    def finished(self) -> bool:
        return self.n_steps >= self.n_total

    def label(self, atoms: Atoms) -> Atoms:
        """Evaluate the teacher and return a detached, labeled copy.

        ``atoms.copy()`` drops ``atoms.calc``, and with it the labels, so the results
        are read out first and reattached as a ``SinglePointCalculator``. Skipping
        this writes geometry with no energy or forces, which is only noticed at
        student-training time. Dropping the calculator is also what discards any
        label the base frame arrived with.
        """
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        labeled = atoms.copy()
        labeled.calc = SinglePointCalculator(labeled, energy=energy, forces=forces)
        return labeled

    def step(self) -> None:
        # Variant-major: every base frame gets variant 0 before any gets variant 1, so
        # a run that stops early has covered all the base frames rather than
        # exhausting the first few.
        base_index = self.n_steps % len(self.base_frames)
        variant_index = self.n_steps // len(self.base_frames)
        self.n_steps += 1

        kind, value = self.variants[variant_index]
        if kind == "iso":
            strain_matrix = isotropic_strain_matrix(value)
        else:
            strain_matrix = random_anisotropic_strain_matrix(
                self.rng, self.max_strain_magnitude
            )

        atoms = self.base_frames[base_index].copy()
        atoms.set_cell(strain_matrix @ atoms.cell[:], scale_atoms=True)
        rattle_positions(self.rng, atoms, self.max_displacement_ang)

        labeled = self.label(atoms)
        labeled.info["base_frame"] = base_index
        self.append(labeled, self.base_frame_split[base_index])
