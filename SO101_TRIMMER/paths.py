"""Canonical locations for everything the project reads and writes.

Why this exists
---------------
Scripts used to find their data with `Path(__file__).parent`, which silently
means "the folder this script happens to live in". That works only while every
file sits in one flat directory — the moment a script moves into a subfolder it
starts looking for the calibration and the model in the wrong place.

Everything now asks THIS module instead, so a file can live anywhere in the
tree and still find the same data.

    from paths import DATA, OUTPUT, CALIBRATION
    cal = json.loads(CALIBRATION.read_text())

Layout
------
    SO101_TRIMMER/
        launch.py          the entry point
        paths.py           this file
        core/              shared modules imported by everything else
        calibrate/         one-off setup routines
        run/               the things that actually drive the robot/camera
        dashboard/         the web app
        training/          dataset building and model training
        tools/             diagnostics and tests
        docs/              documentation
        data/              model, calibration, datasets  (inputs)
        output/            scan sessions and logs         (generated)
"""
from pathlib import Path

# This file sits at the project root, so the root is simply its own folder.
ROOT = Path(__file__).resolve().parent

# --- code ---
CORE = ROOT / 'core'
CALIBRATE = ROOT / 'calibrate'
RUN = ROOT / 'run'
DASHBOARD = ROOT / 'dashboard'
TRAINING = ROOT / 'training'
TOOLS = ROOT / 'tools'
DOCS = ROOT / 'docs'

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
    """Locate a data file, tolerating the pre-reorganisation layout.

    Looks in data/ first, then the project root, then any extra folders given.
    This means an older calibration or model left at the top level still works
    instead of the program simply reporting that it is missing.
    """
    for base in (DATA, ROOT, *extra):
        p = Path(base) / name
        if p.exists():
            return p
    return DATA / name          # canonical location, for a clear error message
