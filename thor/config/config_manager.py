"""YAML-based configuration management for THOR simulation.

Provides frozen dataclasses for each configuration section and
a load_config() function that merges user-supplied YAML with
built-in defaults.  Missing keys fall back to dataclass defaults,
so partial config files are valid.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Section dataclasses (all frozen for immutability at runtime)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    """Simulation parameters."""
    scenario: str = "standing"
    t_final: float = 5.0
    dt: float = 0.002
    walking_speed: float = 0.1875


@dataclass(frozen=True)
class GaitConfig:
    """Gait parameters (Winter 1991)."""
    n_steps: int = 6
    ds_duration: float = 0.25
    swing_duration: float = 0.55


@dataclass(frozen=True)
class ControlConfig:
    """Control gains."""
    kp_leg: float = 600.0
    kd_leg: float = 60.0
    kp_arm: float = 300.0
    kd_arm: float = 30.0


@dataclass(frozen=True)
class CIMPCConfig:
    """CI-MPC parameters."""
    Q_q: float = 500.0
    Q_v: float = 50.0
    R: float = 0.01


@dataclass(frozen=True)
class ContactConfig:
    """Contact model parameters."""
    mu: float = 0.7
    stiffness: float = 30000.0
    damping: float = 2000.0


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization settings."""
    save_plots: bool = True
    save_gif: bool = False
    fps: int = 20
    dpi: int = 300   # Publication quality
    output_dir: str = "output"


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance tuning."""
    use_jit: bool = True
    cholesky_cache_threshold: float = 1e-3
    warmup_on_init: bool = True


@dataclass(frozen=True)
class ThorConfig:
    """Top-level configuration holding all sections."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ci_mpc: CIMPCConfig = field(default_factory=CIMPCConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "simulation": SimulationConfig,
    "gait": GaitConfig,
    "control": ControlConfig,
    "ci_mpc": CIMPCConfig,
    "contact": ContactConfig,
    "visualization": VisualizationConfig,
    "performance": PerformanceConfig,
}


def _merge_dataclass(cls, data: dict):
    """Instantiate *cls* from *data*, ignoring unknown keys.

    Fields absent from *data* receive their dataclass default values,
    so partial YAML sections are valid.
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def _load_yaml_file(path: Path) -> dict:
    """Read a YAML file and return the parsed dict (empty dict on failure)."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> ThorConfig:
    """Load configuration from a YAML file.

    Resolution order
    ----------------
    1. Built-in ``defaults.yaml`` (always loaded first).
    2. User-supplied *path* (if given).
    3. ``config.yaml`` in the project root (searched upward from this file)
       when *path* is ``None`` and no explicit path is provided.

    Keys present in the user file override the defaults; missing keys
    fall back to the dataclass field defaults.

    Args:
        path: Explicit path to a ``.yaml`` configuration file.  Pass
              ``None`` to auto-discover ``config.yaml``.

    Returns:
        ThorConfig: Fully populated, frozen configuration object.
    """
    # Step 1: load built-in defaults
    defaults_path = Path(__file__).parent / "defaults.yaml"
    merged: dict = _load_yaml_file(defaults_path)

    # Step 2: locate and load user config
    if path is not None:
        user_data = _load_yaml_file(Path(path))
    else:
        # Search upward from the package root for config.yaml
        candidate = Path(__file__).parent.parent.parent / "config.yaml"
        user_data = _load_yaml_file(candidate)

    # Step 3: deep-merge user values over defaults (top-level sections only)
    for section, values in user_data.items():
        if isinstance(values, dict):
            merged.setdefault(section, {})
            merged[section].update(values)
        else:
            merged[section] = values

    # Step 4: construct section dataclasses
    kwargs: dict = {}
    for key, cls in _SECTION_MAP.items():
        section_data = merged.get(key, {})
        kwargs[key] = _merge_dataclass(cls, section_data if isinstance(section_data, dict) else {})

    return ThorConfig(**kwargs)
