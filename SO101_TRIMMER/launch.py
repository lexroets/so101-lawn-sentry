"""One-click launcher: live scan window + web app, no tab juggling.

Run THIS (green Run in Thonny) instead of scan_lawn.py. It:
  1. starts the lawn-map web app in the background,
  2. runs the live scan window straight away (no waiting on the web server),
  3. opens the browser automatically the moment the web app is ready.

Press q in the camera window to finish; the web app stays up for review, then
press ENTER in the shell to close it.

    python launch.py
"""
import os
import sys
import socket
import time
import threading
import webbrowser
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / 'lawn_map_app.py'
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


def _start_webapp():
    """Launch Streamlit headless. Returns Popen, or None if reused/unavailable."""
    if _port_open(PORT):
        print(f"[launch] web app already running on {PORT}; reusing it.")
        return None
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


def main():
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
