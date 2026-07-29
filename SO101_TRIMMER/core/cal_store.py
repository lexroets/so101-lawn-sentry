"""Keep every calibration that has ever worked, so a bad one is never fatal.

A calibration took real effort to produce and got overwritten the moment the
next attempt pressed save - including one that was known good. There is no way
to tell from inside a calibration tool whether the result is better than what
it is about to replace, so the tool should not be the thing that decides.

Every save copies the outgoing file into data/calibrations/ first. Nothing is
ever lost, and going back is one command:

    python tools/calibrations.py             # what is there
    python tools/calibrations.py restore 3   # put that one back
    python tools/calibrations.py keep "demo" # bookmark the current one
"""
# --- put the project's folders on sys.path (see paths.py) --------------------
import sys as _sys, pathlib as _pl
_d = _pl.Path(__file__).resolve().parent
while not (_d / 'paths.py').is_file() and _d != _d.parent:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
import paths as _paths
_paths.bootstrap()
# -----------------------------------------------------------------------------
import json
import re
import shutil
from datetime import datetime

from paths import DATA

LIVE = DATA / 'lawn_calibration.json'
STORE = DATA / 'calibrations'


def _slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_') or 'cal'


def archive(label=None, path=None):
    """Copy the current calibration into the store. Returns the new path."""
    src = path or LIVE
    if not src.is_file():
        return None
    STORE.mkdir(parents=True, exist_ok=True)
    try:
        kind = json.loads(src.read_text()).get('calibration_type', 'unknown')
    except (OSError, ValueError):
        kind = 'unreadable'
    stamp = datetime.fromtimestamp(src.stat().st_mtime).strftime('%Y-%m-%d_%H%M')
    name = f'{stamp}_{_slug(label or kind)}.json'
    out = STORE / name
    n = 2
    while out.exists():
        out = STORE / f'{stamp}_{_slug(label or kind)}_{n}.json'
        n += 1
    shutil.copy2(src, out)
    return out


def save(data, label=None):
    """Write a new calibration, archiving whatever it replaces.

    Call this instead of writing lawn_calibration.json directly. The previous
    one survives even if the new one turns out to be nonsense.
    """
    old = archive(label='replaced_' + str(data.get('calibration_type', 'cal')))
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    LIVE.write_text(json.dumps(data, indent=2))
    archive(label=label or data.get('calibration_type'))
    return old


def normalise(cal):
    """Fill in fields a newer calibration method does not produce itself.

    The scan loop wants the lawn's four corners as image pixels
    (`image_points_tl_tr_br_bl`), because the clicking calibrators had them
    lying around. The QR method never clicks anything - it solves the plane
    directly - so that key was simply absent and the scan died on a KeyError
    the instant it started, which showed up as "no video feed" rather than as
    an error.

    The corners are not extra information, though: they are the lawn rectangle
    pushed back through the homography. So derive them rather than demanding
    every calibrator invent them.
    """
    import numpy as np
    if not isinstance(cal, dict) or 'homography_image_to_lawn' not in cal:
        return cal
    if cal.get('image_points_tl_tr_br_bl'):
        return cal
    W = float(cal.get('lawn_width_m', 0) or 0)
    L = float(cal.get('lawn_length_m', 0) or 0)
    if W <= 0 or L <= 0:
        return cal
    try:
        import cv2
        Hi = np.linalg.inv(np.asarray(cal['homography_image_to_lawn'], float))
        pts = cv2.perspectiveTransform(
            np.array([[[0, 0], [W, 0], [W, L], [0, L]]], np.float32),
            Hi.astype(np.float32)).reshape(4, 2)
    except Exception:
        return cal
    if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5:
        return cal
    cal['image_points_tl_tr_br_bl'] = [[float(x), float(y)] for x, y in pts]
    cal.setdefault('reference_frame_points', cal['image_points_tl_tr_br_bl'])
    cal['_corners_derived'] = True
    return cal


def load(path=None):
    """The live calibration, with any missing legacy fields filled in."""
    p = _pl.Path(path) if path else LIVE
    return normalise(json.loads(p.read_text()))


def entries():
    """Everything in the store, newest first, with a one-line summary each."""
    if not STORE.is_dir():
        return []
    out = []
    for p in sorted(STORE.glob('*.json')):
        try:
            j = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({
            'path': p,
            'name': p.name,
            'type': j.get('calibration_type', '?'),
            'lawn': f"{j.get('lawn_width_m','?')} x {j.get('lawn_length_m','?')} m",
            'rest': bool(j.get('rest_pose')),
            'created': j.get('created') or datetime.fromtimestamp(
                p.stat().st_mtime).strftime('%Y-%m-%dT%H:%M:%S'),
            'note': j.get('note', ''),
            'accuracy': j.get('accuracy', {}).get('worst_error_cm')
            or (j.get('marker') or {}).get('worst_error_cm'),
        })
    return sorted(out, key=lambda e: e['created'], reverse=True)


def restore(path):
    """Make an archived calibration the live one (archiving the current first)."""
    p = _pl.Path(path)
    if not p.is_file():
        return False
    archive(label='replaced_by_restore')
    shutil.copy2(p, LIVE)
    return True


def carry_rest_pose(into, frm=None):
    """Lend a rest pose to a calibration that predates the idea of one.

    Only sound if the arm has not been moved between the two, so this is
    offered rather than done automatically - a wrong rest pose puts the plane
    somewhere the camera is not looking, which is the exact failure this whole
    system keeps having.
    """
    try:
        a = json.loads(_pl.Path(into).read_text())
    except (OSError, ValueError):
        return False
    if a.get('rest_pose'):
        return True
    src = frm or LIVE
    try:
        rest = json.loads(_pl.Path(src).read_text()).get('rest_pose')
    except (OSError, ValueError):
        return False
    if not rest:
        return False
    a['rest_pose'] = rest
    a['rest_pose_borrowed_from'] = str(_pl.Path(src).name)
    _pl.Path(into).write_text(json.dumps(a, indent=2))
    return True
