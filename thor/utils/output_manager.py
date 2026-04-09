"""Output directory management for simulation results.

Provides a single OutputManager that owns the directory tree under a
configurable base path (default: ``output/``) and exposes helper
methods for resolving paths and persisting simulation artefacts.

Directory layout
----------------
    <base>/
        plots/    -- PNG/SVG figures
        gifs/     -- animated GIF files
        data/     -- .npz trajectory archives
        logs/     -- text / JSON log files
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import numpy as np


class OutputManager:
    """Manages output directory structure and file I/O.

    All ``*_path`` methods create parent directories on demand so callers
    never need to call ``ensure_dirs()`` explicitly.  The method is still
    available for eager initialisation (e.g., at simulation startup).

    Args:
        base_dir: Root output directory.  Relative paths are resolved
                  relative to the current working directory at call time.
    """

    def __init__(self, base_dir: str = "output") -> None:
        self.base  = Path(base_dir)
        self.plots = self.base / "plots"
        self.gifs  = self.base / "gifs"
        self.data  = self.base / "data"
        self.logs  = self.base / "logs"

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all output sub-directories if they do not already exist."""
        for d in (self.plots, self.gifs, self.data, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers (lazy directory creation)
    # ------------------------------------------------------------------

    def plot_path(self, name: str) -> Path:
        """Return an absolute path under ``plots/`` for *name*.

        Creates ``plots/`` if needed.

        Args:
            name: Filename, e.g. ``"joint_angles.png"``.

        Returns:
            Path object pointing to ``<base>/plots/<name>``.
        """
        self.plots.mkdir(parents=True, exist_ok=True)
        return self.plots / name

    def gif_path(self, name: str) -> Path:
        """Return an absolute path under ``gifs/`` for *name*.

        Creates ``gifs/`` if needed.

        Args:
            name: Filename, e.g. ``"walking_animation.gif"``.

        Returns:
            Path object pointing to ``<base>/gifs/<name>``.
        """
        self.gifs.mkdir(parents=True, exist_ok=True)
        return self.gifs / name

    def data_path(self, name: str) -> Path:
        """Return an absolute path under ``data/`` for *name*.

        Creates ``data/`` if needed.

        Args:
            name: Filename, e.g. ``"trajectory.npz"``.

        Returns:
            Path object pointing to ``<base>/data/<name>``.
        """
        self.data.mkdir(parents=True, exist_ok=True)
        return self.data / name

    def log_path(self, name: str) -> Path:
        """Return an absolute path under ``logs/`` for *name*.

        Creates ``logs/`` if needed.

        Args:
            name: Filename, e.g. ``"run.log"``.

        Returns:
            Path object pointing to ``<base>/logs/<name>``.
        """
        self.logs.mkdir(parents=True, exist_ok=True)
        return self.logs / name

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_trajectory(self, result: dict, name: str) -> Path:
        """Save simulation trajectory arrays to a compressed ``.npz`` file.

        Only ``numpy.ndarray`` values in *result* are saved; scalar and
        non-array entries are silently skipped.

        Args:
            result: Dictionary mapping string keys to ``ndarray`` values.
            name:   Base name (without extension) for the output file.

        Returns:
            Path of the written ``.npz`` file.
        """
        path = self.data_path(f"{name}.npz")
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, **arrays)
        return path

    def save_config_snapshot(self, config, name: str) -> Path:
        """Serialize a ThorConfig (frozen dataclass) to a JSON file.

        Converts the dataclass hierarchy via ``dataclasses.asdict`` and
        writes indented JSON.  Non-serializable values are converted to
        strings via ``default=str``.

        Args:
            config: ThorConfig instance (or any frozen dataclass).
            name:   Base name (without extension) for the output file.

        Returns:
            Path of the written JSON file.
        """
        path = self.data_path(f"{name}_config.json")
        payload = asdict(config)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def timestamped_name(self, prefix: str, ext: str = "") -> str:
        """Generate a timestamped filename component.

        Args:
            prefix: Name prefix, e.g. ``"walking"``.
            ext:    File extension including dot, e.g. ``".png"``.

        Returns:
            String like ``"walking_20260409_153012.png"``.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}{ext}"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"OutputManager(base={self.base!r})"
