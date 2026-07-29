"""ArUco marker detection and printing, across OpenCV versions.

OpenCV moved the aruco API in 4.7 (`cv2.aruco.detectMarkers` became
`ArucoDetector.detectMarkers`) and again renamed the drawing call. Rather than
pin a version — the user may be on anything from a Raspberry Pi image to a
current install — this picks whichever exists.

A marker gives us four coplanar points whose real-world separation we know. Lay
it flat on the grass and it is a direct, sub-pixel measurement of the ground
plane, which is exactly what calibration needs and what clicking a blurry lawn
edge by hand can never be.
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
import cv2
import numpy as np

# 4x4 codes are the coarsest and therefore the most detectable at distance,
# which matters when the marker is 8 m from a security camera.
DICT_NAME = 'DICT_4X4_50'
MARKER_ID = 7

# Paper sizes in metres (width, height).
PAPER = {'a4': (0.210, 0.297), 'a3': (0.297, 0.420),
         'letter': (0.216, 0.279), 'tabloid': (0.279, 0.432)}

# Consumer printers cannot print to the paper edge, and ArUco needs a white
# quiet zone around the code or detection falls apart at distance.
PRINTER_MARGIN_M = 0.006       # unprintable edge, each side
QUIET_ZONE_M = 0.015           # white border around the marker, each side

# The size of the marker ACTUALLY IN USE, measured off the print with a tape -
# not the size it was sent to the printer at. Printers shrink to fit and rarely
# mention it: the A3 sheet was laid out at 25.5 cm and came out at 21.5 cm.
# Every distance this system reports is scaled by this number, so a 16% error
# here is a 16% error in the lawn. Measure, do not assume.
DEFAULT_SIZE_M = 0.215


def max_size_for(paper='a4'):
    """Largest marker side that will actually print on this paper."""
    w, _ = PAPER.get(paper, PAPER['a4'])
    return w - 2 * (PRINTER_MARGIN_M + QUIET_ZONE_M)


def _dictionary():
    a = cv2.aruco
    d = getattr(a, DICT_NAME)
    if hasattr(a, 'getPredefinedDictionary'):
        return a.getPredefinedDictionary(d)
    return a.Dictionary_get(d)                     # OpenCV < 4.7


def _detector_params():
    a = cv2.aruco
    if hasattr(a, 'DetectorParameters'):
        try:
            return a.DetectorParameters()
        except TypeError:
            pass
    if hasattr(a, 'DetectorParameters_create'):
        return a.DetectorParameters_create()
    return None


def _tune(p):
    """Favour detection at distance and on grass."""
    if p is None:
        return p
    try:
        # sub-pixel corner refinement is what makes the calibration accurate
        p.cornerRefinementMethod = getattr(cv2.aruco, 'CORNER_REFINE_SUBPIX', 1)
        p.cornerRefinementWinSize = 5
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 33
        p.adaptiveThreshWinSizeStep = 6
        p.minMarkerPerimeterRate = 0.01   # allow a small, distant marker
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
    except Exception:
        pass
    return p


class MarkerFinder:
    """Finds our marker in a frame. `find(frame)` -> 4x2 corners or None."""

    def __init__(self, marker_id=MARKER_ID):
        self.marker_id = marker_id
        self.dict = _dictionary()
        self.params = _tune(_detector_params())
        self._obj = None
        if hasattr(cv2.aruco, 'ArucoDetector'):        # OpenCV >= 4.7
            self._obj = cv2.aruco.ArucoDetector(self.dict, self.params)

    def detect_all(self, frame):
        """-> {id: 4x2 float32 corners} for every marker seen."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._obj is not None:
            corners, ids, _ = self._obj.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dict, parameters=self.params)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = np.asarray(c, np.float32).reshape(4, 2)
        return out

    def find(self, frame):
        """Our marker specifically, or any single marker if ours isn't there."""
        found = self.detect_all(frame)
        if self.marker_id in found:
            return found[self.marker_id]
        if len(found) == 1:            # be forgiving: they printed a different id
            return next(iter(found.values()))
        return None


def marker_image(size_px=800, marker_id=MARKER_ID):
    a = cv2.aruco
    d = _dictionary()
    if hasattr(a, 'generateImageMarker'):
        return a.generateImageMarker(d, marker_id, size_px)
    return a.drawMarker(d, marker_id, size_px)      # OpenCV < 4.7


def printable_sheet(side_m=DEFAULT_SIZE_M, marker_id=MARKER_ID, dpi=150,
                    paper='a4'):
    """A sheet with the marker at a known printed size, plus instructions.

    Printing at 100% scale (NOT 'fit to page') is what makes the printed size
    match the number we calibrate against. The sheet says so, and includes
    witness marks so it can be checked with a tape measure.

    Returns (sheet, actual_side_m) — the actual size may be smaller than asked
    if it would not fit the paper, and the caller must tell the user.
    """
    pw, ph = PAPER.get(paper, PAPER['a4'])
    px_per_m = dpi / 0.0254
    W = int(round(pw * px_per_m))
    H = int(round(ph * px_per_m))
    sheet = np.full((H, W, 3), 255, np.uint8)

    fits = max_size_for(paper)
    if side_m > fits:
        side_m = fits                       # caller reports this
    side_px = int(round(side_m * px_per_m))

    m = marker_image(side_px, marker_id)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    x0 = (W - side_px) // 2
    y0 = int(1.9 * dpi)
    sheet[y0:y0 + side_px, x0:x0 + side_px] = m

    F = cv2.FONT_HERSHEY_SIMPLEX

    def text(s, y, scale=0.62, thick=1, colour=(30, 30, 30), centre=True):
        (tw, _), _ = cv2.getTextSize(s, F, scale, thick)
        x = (W - tw) // 2 if centre else int(0.6 * dpi)
        cv2.putText(sheet, s, (x, y), F, scale, colour, thick, cv2.LINE_AA)

    text('LAWN SENTRY  -  GROUND CALIBRATION MARKER', int(0.85 * dpi), 0.85, 2, (20, 20, 20))
    text('Print at 100% scale.  Do NOT use "fit to page" or "shrink to fit".',
         int(1.25 * dpi), 0.6, 1, (0, 0, 190))
    text(f'Printed square must measure {side_m * 100:.1f} cm on each side.',
         int(1.55 * dpi), 0.6, 1)

    # size witness marks along the bottom edge of the marker
    cv2.line(sheet, (x0, y0 + side_px + 18), (x0 + side_px, y0 + side_px + 18),
             (0, 0, 190), 2, cv2.LINE_AA)
    for xx in (x0, x0 + side_px):
        cv2.line(sheet, (xx, y0 + side_px + 8), (xx, y0 + side_px + 28),
                 (0, 0, 190), 2, cv2.LINE_AA)
    text(f'{side_m * 100:.1f} cm  - check this with a tape measure',
         y0 + side_px + 52, 0.55, 1, (0, 0, 190))

    lines = [
        'HOW TO USE',
        '1. Glue this sheet to card so it lies perfectly flat. Keep the white',
        '   border - detection needs it.',
        '2. Lay it FLAT ON THE GROUND at one corner of the area you want',
        '   watched, fully visible to the camera.',
        '3. Run the calibration. Move the marker to each corner in turn,',
        '   going around the edge - the order matters, so do not skip about.',
        '   The FIRST corner you place becomes the map origin.',
        '4. Type the real width and length of the area when asked.',
        '',
        'TIPS',
        '- Lay it flat. A tilted or curled marker is the main source of error.',
        '- Weigh the corners down on windy days.',
        '- Not detected? Better light, or print it larger (--paper a3).',
        '- Big garden? The marker must still be ~30 px on camera at the far',
        '  corner, so a large area needs a large marker.',
    ]
    # fit the block into whatever space is left above the footer
    top = y0 + side_px + int(0.55 * dpi)
    bottom = H - int(0.6 * dpi)
    step = max(14, int((bottom - top) / max(len(lines), 1)))
    y = top
    for ln in lines:
        bold = ln in ('HOW TO USE', 'TIPS')
        text(ln, y, 0.6 if bold else 0.52, 2 if bold else 1,
             (20, 20, 20) if bold else (60, 60, 60), centre=False)
        y += step

    cv2.putText(sheet, f'marker id {marker_id}  -  {DICT_NAME}  -  '
                f'{side_m*100:.1f} cm  -  {paper.upper()}',
                (int(0.6 * dpi), H - int(0.45 * dpi)), F, 0.45, (150, 150, 150), 1,
                cv2.LINE_AA)
    return sheet, side_m
