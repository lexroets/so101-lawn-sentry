"""Preflight self-check — run BEFORE enabling torque.

Catches the silent misconfigurations that otherwise only show up as weird
behaviour after the arm is already moving: a missing calibration, an absent
model file, the camera unplugged, the wrong serial port. Prints a clear
GO / NO-GO report and returns (ok, report).

Pure inspection — never opens the serial port, never enables torque.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# (attr_name, human label, critical?)  — checked against the config module.
_CAL_FILES = [
    ('reach_calibration.py', 'arm reach poses', True),
    ('base_calibration.py', 'base pixel/latency', True),
    ('gripper_calibration.py', 'gripper open/closed', True),
    ('buckets_calibration.py', 'two-bucket sort', True),
    ('lying_calibration.py', 'top-down lying grab', False),
    ('colour_calibration.py', 'colour thresholds', False),
    ('roll_calibration.py', 'auto wrist-roll', False),
    ('detection_calibration.py', 'battery sizes', False),
]

_OK, _WARN, _BAD = ' OK ', 'WARN', 'FAIL'   # plain text: Thonny shows raw
                                            # ANSI codes as garbage.


def _line(state, label, detail=''):
    return f'  [{state}] {label}' + (f'  -- {detail}' if detail else '')


def check(require_model=False):
    """require_model=True for the ML build (needs battery_cls.onnx)."""
    lines, ok = [], True

    # --- config imports cleanly? -----------------------------------------
    try:
        import config as cfg
        lines.append(_line(_OK, 'config.py loads'))
    except Exception as exc:
        return False, [_line(_BAD, 'config.py FAILED to import', str(exc))]

    # --- serial port ------------------------------------------------------
    port = getattr(cfg, 'SERIAL_PORT', None)
    if port and os.path.exists(port):
        lines.append(_line(_OK, 'serial port', port))
    else:
        ok = False
        lines.append(_line(_BAD, 'serial port MISSING', str(port)
                           + ' — is the arm plugged in / powered?'))

    # --- camera present? --------------------------------------------------
    try:
        import depthai as dai
        devs = dai.Device.getAllAvailableDevices()
        if devs:
            lines.append(_line(_OK, 'OAK camera', f'{len(devs)} device(s)'))
        else:
            ok = False
            lines.append(_line(_BAD, 'OAK camera NOT found',
                               'check the USB-C cable'))
    except Exception as exc:
        lines.append(_line(_WARN, 'camera check skipped', str(exc)))

    # --- calibration files ------------------------------------------------
    for fname, label, critical in _CAL_FILES:
        present = os.path.isfile(os.path.join(_HERE, fname))
        if present:
            lines.append(_line(_OK, label, fname))
        elif critical:
            ok = False
            lines.append(_line(_BAD, f'{label} MISSING', fname
                               + ' — run its calibrate_*.py'))
        else:
            lines.append(_line(_WARN, f'{label} not calibrated', fname))

    # --- config sanity ----------------------------------------------------
    if getattr(cfg, 'BUCKET_GREEN', None) and getattr(cfg, 'BUCKET_RED', None):
        lines.append(_line(_OK, 'buckets defined'))
    else:
        ok = False
        lines.append(_line(_BAD, 'buckets undefined',
                           'run calibrate_buckets.py'))
    if getattr(cfg, 'REACH_POINTS', None):
        lines.append(_line(_OK, 'reach table', f"{len(cfg.REACH_POINTS)} pts"))
    else:
        ok = False
        lines.append(_line(_BAD, 'reach table empty'))

    # --- ML model (only for main_ml) -------------------------------------
    if require_model:
        onnx = any(os.path.isfile(p) for p in (
            os.path.join(_HERE, 'battery_cls.onnx'),
            os.path.join(os.path.dirname(_HERE), 'ml_training',
                         'battery_cls.onnx')))
        labels = os.path.isfile(os.path.join(_HERE, 'labels.txt'))
        if onnx and labels:
            lines.append(_line(_OK, 'ML model + labels'))
        elif onnx:
            lines.append(_line(_WARN, 'labels.txt missing',
                               'assuming alphabetical class order'))
        else:
            ok = False
            lines.append(_line(_BAD, 'battery_cls.onnx MISSING',
                               'train it, or run main.py instead'))

    return ok, lines


def report(require_model=False, hard=False):
    """Print the report. If hard=True, a NO-GO raises SystemExit."""
    ok, lines = check(require_model)
    print('\n' + '=' * 62)
    print(' PREFLIGHT  ' + ('GO' if ok else 'NO-GO'))
    print('=' * 62)
    for ln in lines:
        print(ln)
    print('=' * 62)
    if not ok:
        print(' Fix the FAIL rows above before running. WARN rows are '
              'optional features.')
        if hard:
            raise SystemExit(1)
    print()
    return ok


if __name__ == '__main__':
    report(require_model=os.path.isfile(
        os.path.join(_HERE, 'battery_cls.onnx')))
