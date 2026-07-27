"""Start the whole demo: web dashboard + live scan, in one command.

    python launch.py
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
import os
import sys
import socket
import time
import threading
import webbrowser
import subprocess
from pathlib import Path

from paths import ROOT, DATA, DASHBOARD, CALIBRATE, ensure

ensure()
HERE = ROOT
APP = DASHBOARD / 'lawn_map_app.py'
PORT = 8501
URL = f'http://localhost:{PORT}'


def _port_open(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _have_streamlit():
    try:
        import streamlit  # noqa: F401
        return True
    except ImportError:
        return False


ACT_PORT = 8799


def _running_build():
    """Ask the running web app which version of itself it is, or None."""
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen(
                f'http://127.0.0.1:{ACT_PORT}/version', timeout=1.5) as r:
            return _json.loads(r.read()).get('build')
    except Exception:
        return None


def _expected_build():
    import re
    try:
        m = re.search(r"^APP_BUILD\s*=\s*'([^']+)'",
                      APP.read_text(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def _kill_port(port):
    """Stop whatever holds this port. Returns True if the port came free."""
    try:
        out = subprocess.run(['lsof', '-ti', f'tcp:{port}'],
                             capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return False
    pids = [p for p in out.split() if p.strip().isdigit()]
    if not pids:
        return False
    for sig in ('-TERM', '-KILL'):
        for p in pids:
            try:
                subprocess.run(['kill', sig, p], capture_output=True, timeout=5)
            except Exception:
                pass
        for _ in range(12):
            if not _port_open(port):
                return True
            time.sleep(0.25)
    return not _port_open(port)


def _start_webapp():
    """Launch Streamlit headless. Returns Popen, or None if reused/unavailable."""
    if _port_open(PORT):
        want, have = _expected_build(), _running_build()
        if want and have == want:
            print(f"[launch] web app already running on {PORT}; reusing it.")
            return None
        # A web app started before the current code exists is the single most
        # confusing failure mode there is: the browser keeps showing an old
        # dashboard and every fix looks like it did nothing. Replace it.
        print(f"[launch] a web app is running on {PORT} but it is an OLD build "
              f"({have or 'pre-versioning'}); restarting it so you get the "
              f"current dashboard.")
        if not _kill_port(PORT):
            print(f"[launch] could not stop it automatically. Close that "
                  f"terminal/tab, or run:  lsof -ti tcp:{PORT} | xargs kill")
            return None
        _kill_port(ACT_PORT)          # its data endpoint too, else the port sticks
        time.sleep(0.6)
    if not _have_streamlit():
        print("\n[launch] Streamlit is NOT installed in this Python, so the web "
              "app can't open.\n"
              "         The scan will still run. To get the map, install once:\n"
              "             pip install streamlit pandas streamlit-autorefresh\n"
              "         (in Thonny: Tools > Manage packages > search 'streamlit')\n")
        return None
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(APP),
           '--server.port', str(PORT), '--server.headless', 'true']
    print("[launch] starting the web app in the background...")
    try:
        return subprocess.Popen(cmd, cwd=str(HERE), env=env)
    except Exception as exc:
        print(f"[launch] could not start the web app: {exc}")
        return None


def _open_browser_when_ready(timeout=45.0):
    """Background thread: open the browser as soon as the server answers."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if _port_open(PORT):
            try:
                webbrowser.open(URL)
            except Exception:
                pass
            print(f"[launch] web app is live -> {URL}")
            return
        time.sleep(0.5)
    print(f"[launch] web app slow to start — open {URL} in your browser manually.")


def _preflight():
    """Quick demo readiness report — say what mode we're in before anything runs."""
    print('\n--- SO-101 Lawn Sentry ---')
    cal = DATA / 'lawn_calibration.json'
    if not cal.is_file():
        print('[launch] NOT CALIBRATED — run:  python calibrate/calibrate_lawn_scan.py')
        return False
    model = DATA / 'poop_yolo.onnx'
    if model.is_file() and model.stat().st_size > 10_000:
        print(f'[launch] detector : TRAINED MODEL ({model.stat().st_size/1e6:.1f} MB)')
    else:
        print('[launch] detector : colour/box heuristic (no trained model yet)')
        print('          -> after training:  python training/deploy_model.py')
    try:
        import poop_config as cfg
        print(f'[launch] lawn     : calibrated, conf threshold '
              f'{getattr(cfg, "POOP_YOLO_CONF", "n/a")}')
    except Exception:
        pass
    print('---------------------------\n')
    return True


def main():
    if not _preflight():
        return
    proc = _start_webapp()
    if proc is not None or _port_open(PORT):
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    # Start the live scan immediately so the camera window appears right away.
    print("[launch] opening the scan window — press q there to finish.\n")
    try:
        import scan_lawn
        scan_lawn.main()
    except SystemExit as exc:               # scan_lawn exits if not calibrated
        print(f"[launch] {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            print("\n[launch] scan finished — the web app is still open for review.")
            try:
                input("[launch] press ENTER to close the web app and quit... ")
            except EOFError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[launch] done.")
        

if __name__ == '__main__':
    main()
