# Set 2026-07-21 from 43 in-situ samples (calibrate_colour.py).
#
# WHAT THE DATA SHOWED:
#   DARKNESS separates cleanly — Ryobi 539-2215 dark px, pink 6-61. ~10x gap.
#   HUE is useless          — Ryobi hue 80-88, pink hue 88-107, sat 20-50
#                             for both. The pink cells are too pale to carry
#                             a stable hue, so pink_px was 0 on every sample.
#   LENGTH is useless       — Ryobi measured 29..99mm, pink 29..93mm. The
#                             detector often measures a fragment.
#
# So the sort now runs on darkness alone, which is also immune to the green
# mat (not dark) and to orientation (a lying black battery is still black).
DARK_SPLIT_FRAC = 0.15
USE_SIZE_OVERRIDE = False
LIME_AS_RYOBI = False
RYOBI_MIN_LEN_M = 0.090
PINK_VS_RYOBI_W = 0.45
