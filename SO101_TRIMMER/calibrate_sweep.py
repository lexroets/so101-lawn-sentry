"""Teach the side-to-side SWEEP range — the base yaw limits the scan covers.

Torque is released so you can rotate the base by hand. Move it to the LEFT end
of the arc you want scanned and press ENTER, then to the RIGHT end and press
ENTER. Optionally set how many stations (stops) to sample across that arc.

Writes poop_scan_range.py (SCAN_BASE_MIN_RAW / SCAN_BASE_MAX_RAW /
SCAN_STATIONS), which poop_config.py auto-loads. Re-run any time to change it.

    python calibrate_sweep.py
"""
from trimmer_arm import TrimmerArm, TrimmerError
from poop_config import (BASE_ID, BASE_RAW_MIN, BASE_RAW_MAX, BASE_FORWARD_RAW,
                         RAW_PER_DEG, SCAN_STATIONS)


def _deg(raw):
    return (raw - BASE_FORWARD_RAW) / RAW_PER_DEG


def main():
    try:
        arm = TrimmerArm()
    except TrimmerError as exc:
        print(exc)
        return
    try:
        arm.torque_off()
        print("Torque OFF — you can turn the base by hand.\n")
        input("Move the base to the LEFT end of the sweep, then press ENTER...")
        a = arm.read_position(BASE_ID)
        input("Now move it to the RIGHT end of the sweep, then press ENTER...")
        b = arm.read_position(BASE_ID)

        lo, hi = sorted((a, b))
        lo = max(BASE_RAW_MIN, lo)
        hi = min(BASE_RAW_MAX, hi)
        if hi - lo < 20:
            print(f"Those two positions are basically the same ({lo}..{hi}). "
                  "Re-run and move the base further between captures.")
            return

        span_deg = (hi - lo) / RAW_PER_DEG
        default_n = max(2, min(31, int(round(span_deg / 8)) + 1))  # ~1 stop / 8 deg
        try:
            raw = input(f"\nStations across the arc [{default_n}]: ").strip()
            n = int(raw) if raw else default_n
        except ValueError:
            n = default_n
        n = max(2, min(41, n))

        with open('poop_scan_range.py', 'w') as f:
            f.write('"""Taught base sweep range. Regenerate with '
                    'calibrate_sweep.py."""\n')
            f.write(f'SCAN_BASE_MIN_RAW = {lo}\n')
            f.write(f'SCAN_BASE_MAX_RAW = {hi}\n')
            f.write(f'SCAN_STATIONS = {n}\n')

        print(f"\nSaved sweep: {lo}..{hi} raw  "
              f"({_deg(lo):+.0f} deg to {_deg(hi):+.0f} deg, "
              f"{span_deg:.0f} deg wide), {n} stations -> poop_scan_range.py")
        print("Run:  python scan_poop.py")
    finally:
        arm.close()


if __name__ == '__main__':
    main()
