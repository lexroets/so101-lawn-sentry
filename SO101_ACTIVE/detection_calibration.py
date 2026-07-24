# Curated 2026-07-20 — the auto-measured values were INVALID: both hue
# ranges came out identical (25,35) because the sampling was dominated by
# desk background, and the size gates rejected a lying-down battery.
# These values are physically derived for the Ryobi RB4L30 (~115x40x23 mm)
# plus a standard pink band. calibrate_all.py Part B is fixed (inset
# sampling + sanity checks) if you want measured values again.
# Widened 2026-07-20: size_rej was running 22-94 per frame with batteries in
# plain view. A battery seen at an angle is foreshortened, and the pink cells
# are ~30% smaller than the RB4L30, so the band has to cover both.
BATT_MIN_DIM_M = (0.014, 0.058)
BATT_MAX_DIM_M = (0.028, 0.140)
RYOBI_HUE = (22, 48)      # lime green-yellow (covers the measured 25-35)
PINK_HUE = (140, 178)     # magenta-pink
