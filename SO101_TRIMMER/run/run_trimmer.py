"""Replay the taught trimmer sweep smoothly, on a loop.

Phase 1 of the grass-trimmer project: get the recalibrated 4-joint arm moving
naturally with the new attachment. No camera yet — this just proves the arm
and the taught motion. The autonomous grass-seeking layer (which reuses the
battery system's camera + base-centring framework) comes next.

    python run_trimmer.py            # sweep forever (Ctrl-C to stop)
    python run_trimmer.py 5          # sweep 5 times, then park at REST

Motion is BLENDED: the next waypoint is commanded once the arm is within
BLEND_TOL of the current one, so the sweep flows continuously instead of
stopping at every point. Tune SWEEP_SPEED / SWEEP_ACC / BLEND_TOL_RAW in
trimmer_config.py to taste.
"""

# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import sys
import time

from trimmer_config import (
    REST_POSE, SWEEP, SWEEP_SPEED, SWEEP_ACC, MOVE_SPEED, MOVE_ACC,
    BLEND_TOL_RAW, ARRIVE_TOL_RAW, SETTLE_S,
)
from trimmer_arm import TrimmerArm, TrimmerError


def blend_through(arm, waypoints, speed, acc):
    """Move through waypoints, starting each next move when within BLEND_TOL
    of the current target — continuous, non-stop motion."""
    arm.set_profile(speed, acc)
    for i, wp in enumerate(waypoints):
        arm.move_pose(wp)
        # Last point of a pass settles fully; intermediate points blend.
        tol = ARRIVE_TOL_RAW if i == len(waypoints) - 1 else BLEND_TOL_RAW
        arm.wait_arrived(wp, tol=tol)


def main():
    loops = None
    if len(sys.argv) > 1:
        try:
            loops = int(sys.argv[1])
        except ValueError:
            pass

    if REST_POSE is None or not SWEEP:
        print("No taught poses. Run  python calibrate_trimmer.py  first.")
        return

    arm = TrimmerArm()
    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    arm.enable()

    # A cutting PASS is the sweep out, then reversed back to the start, so the
    # trimmer works the patch both directions and returns near REST cleanly.
    forward = list(SWEEP)
    backward = list(reversed(SWEEP))

    try:
        print("Moving to REST...")
        arm.set_profile(MOVE_SPEED, MOVE_ACC)
        arm.move_pose(REST_POSE)
        arm.wait_arrived(REST_POSE)
        time.sleep(SETTLE_S)

        n = 0
        while loops is None or n < loops:
            n += 1
            print(f"sweep {n}" + (f"/{loops}" if loops else ""))
            blend_through(arm, forward, SWEEP_SPEED, SWEEP_ACC)
            blend_through(arm, backward, SWEEP_SPEED, SWEEP_ACC)
            # Lift clear of the grass between passes.
            arm.set_profile(MOVE_SPEED, MOVE_ACC)
            arm.move_pose(REST_POSE)
            arm.wait_arrived(REST_POSE, tol=BLEND_TOL_RAW)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        try:
            arm.set_profile(MOVE_SPEED, MOVE_ACC)
            arm.move_pose(REST_POSE)
            arm.wait_arrived(REST_POSE)
            time.sleep(0.5)
            arm.torque_off()
        except TrimmerError:
            pass
        arm.close()
        print("parked at REST, torque off.")


if __name__ == '__main__':
    main()
