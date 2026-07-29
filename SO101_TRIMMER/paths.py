"""Canonical locations for everything the project reads and writes.

Why this exists
---------------
Scripts used to find their data with `Path(__file__).parent`, which silently
means "the folder this script happens to live in". That works only while every
file sits in one flat directory - the moment a script moves into a subfolder it
starts looking for the calibration and the model in the wrong place.

Everything asks THIS module instead, so a file can live anywhere in the tree
and still find the same data.

    from paths import DATA, OUTPUT, CALIBRATION
    cal = json.loads(CALIBRATION.read_text())

Layout
------
    SO101_TRIMMER/
        launch.py          start the Lawn Sentry demo
        paths.py           this file
        core/              shared: arm, camera, ground geometry, markers
        lawn_sentry/       the poop-detection product
            calibrate/     set the lawn plane and the arm's rest pose
        trimmer/           the earlier grass-trimming robot
            calibrate/     its own setup routines
        training/          dataset building and model training
        tools/             diagnostics and one-off fixes
        archive/           superseded, kept for reference, imported by nothing
        docs/              documentation
        data/              model, calibration, datasets   (inputs)
        output/            scan sessions and logs         (generated)

Two projects share this tree because they share the hardware: the same SO-101
arm, the same OAK-D, the same serial protocol. Splitting core/ in two would
mean maintaining two copies of the arm driver.

The import bootstrap
--------------------
Every module starts with a short block that puts the project's folders on
sys.path, so `import poop_config` works from any depth without packaging the
whole thing. That block used to name the folders explicitly, which meant
moving a file to a new folder broke every import in it. It now DISCOVERS the
folders, so the layout above can be rearranged freely.
"""
from pathlib import Path
import sys as _sys

# This file sits at the project root, so the root is simply its own folder.
ROOT = Path(__file__).resolve().parent

# Folders that hold data rather than code - never added to the import path.
# archive/ is excluded deliberately: it holds superseded versions of modules
# that are still live under other names, and having both importable invites the
# wrong one being picked up. Scripts in there still run directly, because
# Python adds a script's own folder for you.
_NOT_CODE = {'data', 'output', 'docs', 'archive', '__pycache__', '.git',
             '.streamlit', 'node_modules', '.venv', 'venv'}


def code_dirs(root=None):
    """Every folder under the root that contains Python worth importing."""
    base = Path(root or ROOT)
    out = [base]
    for p in sorted(base.rglob('*')):
        if not p.is_dir():
            continue
        parts = set(p.relative_to(base).parts)
        if parts & _NOT_CODE or any(s.startswith('.') for s in parts):
            continue
        if any(f.suffix == '.py' for f in p.iterdir() if f.is_file()):
            out.append(p)
    return out


def bootstrap(root=None):
    """Put the project's code folders on sys.path. Safe to call repeatedly."""
    for d in code_dirs(root):
        s = str(d)
        if s not in _sys.path:
            _sys.path.insert(0, s)


# --- code ---
CORE = ROOT / 'core'
LAWN = ROOT / 'lawn_sentry'
CALIBRATE = LAWN / 'calibrate'
TRIMMER = ROOT / 'trimmer'
TRAINING = ROOT / 'training'
TOOLS = ROOT / 'tools'
ARCHIVE = ROOT / 'archive'
DOCS = ROOT / 'docs'
# kept so anything still asking for the old names keeps working
RUN = LAWN
DASHBOARD = LAWN

# --- data in ---
DATA = ROOT / 'data'
CALIBRATION = DATA / 'lawn_calibration.json'
LAWN_EXTENT = DATA / 'lawn_extent.json'
MODEL_DIR = DATA

# --- data out ---
OUTPUT = ROOT / 'output'
SESSIONS = OUTPUT / 'scan_sessions'
POOP_LOGS = OUTPUT / 'poop_logs'


def ensure():
    """Create the writable folders if they don't exist yet."""
    for d in (DATA, OUTPUT, SESSIONS, POOP_LOGS):
        d.mkdir(parents=True, exist_ok=True)


def find(name, *extra):
    """Locate a data file, tolerating older layouts.

    Looks in data/ first, then the project root, then any extra folders given.
    This means an older calibration or model left at the top level still works
    instead of the program simply reporting that it is missing.
    """
    for base in (DATA, ROOT, *extra):
        p = Path(base) / name
        if p.exists():
            return p
    return DATA / name          # canonical location, for a clear error message
