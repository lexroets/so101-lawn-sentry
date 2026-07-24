import numpy as np
from config import *


def propose_base_step(current_raw, pixel_error):
    """Return a bounded step toward centring the target."""
    if abs(pixel_error) <= IMAGE_DEADBAND_PX:
        return current_raw, "centred"

    direction = 1 if pixel_error > 0 else -1
    magnitude = min(
        MAX_PROPOSED_STEP_RAW,
        max(1, int(abs(pixel_error) / 18)),
    )
    proposed = current_raw + direction * magnitude

    if not BASE_RAW_MIN <= proposed <= BASE_RAW_MAX:
        raise ValueError("proposed base step violates verified raw limits")

    return proposed, "proposed"
