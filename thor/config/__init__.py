"""Configuration package for THOR simulation.

Public exports
--------------
ThorConfig  -- top-level frozen dataclass with all sections.
load_config -- factory that merges YAML file with built-in defaults.
"""

from .config_manager import (
    ThorConfig,
    SimulationConfig,
    GaitConfig,
    ControlConfig,
    CIMPCConfig,
    ContactConfig,
    VisualizationConfig,
    PerformanceConfig,
    load_config,
)

__all__ = [
    "ThorConfig",
    "SimulationConfig",
    "GaitConfig",
    "ControlConfig",
    "CIMPCConfig",
    "ContactConfig",
    "VisualizationConfig",
    "PerformanceConfig",
    "load_config",
]
