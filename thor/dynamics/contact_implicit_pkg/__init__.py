"""Contact-implicit dynamics package. Re-exports for backward compatibility."""
from .time_stepping import contact_implicit_step
from .simulation import run_contact_implicit_simulation

__all__ = ["contact_implicit_step", "run_contact_implicit_simulation"]
