"""
Gait phase detector.

Determines the current gait phase (double support, left/right swing)
based on elapsed time and gait timing parameters.
"""


def detect_phase(
    t: float,
    ds_duration: float,
    swing_duration: float,
    n_steps: int,
) -> dict:
    """Determine gait phase at time t.

    Returns dict with:
        phase: "ds" | "swing_left" | "swing_right"
        step: step index (-1 for initial DS)
        s: normalized phase progress [0, 1]
    """
    if t < ds_duration:
        return {"phase": "ds", "step": -1, "s": t / ds_duration}

    t_rel = t - ds_duration
    step_cycle = swing_duration + ds_duration

    step_idx = int(t_rel / step_cycle)
    t_in_cycle = t_rel - step_idx * step_cycle

    if step_idx >= n_steps:
        return {"phase": "ds", "step": n_steps, "s": 1.0}

    if t_in_cycle < swing_duration:
        is_left = (step_idx % 2 == 0)
        s = t_in_cycle / swing_duration
        return {
            "phase": "swing_left" if is_left else "swing_right",
            "step": step_idx,
            "s": s,
        }
    else:
        s = (t_in_cycle - swing_duration) / ds_duration
        return {"phase": "ds", "step": step_idx, "s": s}
