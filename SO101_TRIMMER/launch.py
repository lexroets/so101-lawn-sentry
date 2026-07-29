"""Start the whole demo: web dashboard + live scan, in one command.

    python launch.py
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


def _free_port(port):
    """Kill whatever holds this port. Always. No exceptions, no cleverness.

    Trying to reuse an already-running web app was a mistake: a Streamlit
    started before the code changed keeps serving the OLD page, so every fix
    looks like it did nothing and the browser shows a dashboard that no longer
    matches the files on disk. Starting clean every time costs two seconds and
    removes that entire class of confusion.
    """
    if not _port_open(port):
        return True
    for tool in (['lsof', '-ti', f'tcp:{port}'],
                 ['fuser', '-n', 'tcp', str(port)]):
        try:
            out = subprocess.run(tool, capture_output=True, text=True,
                                 timeout=6).stdout
        except Exception:
            continue
        pids = [p for p in out.replace('/tcp:', ' ').split() if p.isdigit()]
        if not pids:
            continue
        print(f"[launch] stopping the old process on port {port} "
              f"(pid {', '.join(pids)})")
        for sig in ('-TERM', '-KILL'):
            for p in pids:
                try:
                    subprocess.run(['kill', sig, p], capture_output=True, timeout=5)
                except Exception:
                    pass
            for _ in range(16):
                if not _port_open(port):
                    return True
                time.sleep(0.25)
    return not _port_open(port)


def _start_webapp():
    """Always start a FRESH web app. Never reuse whatever was already there."""
    if not _have_streamlit():
        print("\n[launch] Streamlit is NOT installed in this Python, so the web "
              "app can't open.\n"
              "         The scan will still run. To get the map, install once:\n"
              "             pip install streamlit pandas streamlit-autorefresh\n")
        return None
    if not APP.is_file():
        print(f"\n[launch] the dashboard is missing:\n         {APP}\n")
        return None

    for port in (PORT, ACT_PORT):
        if _port_open(port) and not _free_port(port):
            print(f"[launch] could not free port {port}. Close it yourself with:")
            print(f"             lsof -ti tcp:{port} | xargs kill")
            print("         then run launch.py again.")
            return None
    time.sleep(0.4)

    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(APP),
           '--server.port', str(PORT), '--server.headless', 'true',
           # THE cause of the laggy video. The scan writes live.jpg ~15 times a
           # second into this folder tree; Streamlit's file watcher sees those
           # changes and RERUNS THE SCRIPT - measured at every 2.6 s in the
           # logs. Each rerun rebuilds the component iframe, which tears down
           # the video connection and restarts it. Nothing in the app can be
           # smooth while that is happening.
           '--server.fileWatcherType', 'none',
           '--server.runOnSave', 'false',
           # the app is one static page; these just add work
           '--client.toolbarMode', 'minimal',
           '--browser.gatherUsageStats', 'false']
    print(f"[launch] starting the web app: {APP.name}")
    try:
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env)
    except Exception as exc:
        print(f"[launch] could not start the web app: {exc}")
        return None

    # confirm it really came up, and say so plainly if it did not
    for _ in range(60):
        if proc.poll() is not None:
            print(f"[launch] the web app EXITED immediately (code {proc.returncode}).")
            print("         Run it directly to see the error:")
            print(f"             python -m streamlit run {APP}")
            return None
        if _port_open(PORT):
            print(f"[launch] web app is up on {PORT}")
            return proc
        time.sleep(0.5)
    print("[launch] the web app is slow to start; carrying on.")
    return proc


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
        print('[launch] NOT CALIBRATED — run:  python lawn_sentry/calibrate/calibrate_lawn_scan.py')
        return False
    # Say plainly how old the calibration is. A homography describes ONE camera
    # pose, so if the camera has been remounted since this was measured the map
    # will be wrong and nothing downstream can detect it.
    import json as _j, time as _t, datetime as _dt
    # Check the calibration has everything the scan needs BEFORE starting
    # anything. A missing field used to surface as a silent crash behind the
    # web app's "press ENTER" prompt, which reads as "the camera is broken".
    try:
        import cal_store
        _chk = cal_store.load(cal)
        _missing = [k for k in ('homography_image_to_lawn', 'lawn_width_m',
                                'lawn_length_m', 'image_points_tl_tr_br_bl')
                    if not _chk.get(k)]
        if _missing:
            print(f"[launch] the calibration is missing: {', '.join(_missing)}")
            print('         Recalibrate:  python lawn_sentry/calibrate/calibrate_qr.py')
            return False
    except Exception as _e:
        print(f'[launch] could not read the calibration: {_e}')
        return False
    try:
        _c = _j.loads(cal.read_text())
        _age_h = (_t.time() - cal.stat().st_mtime) / 3600.0
        _when = _dt.datetime.fromtimestamp(cal.stat().st_mtime).strftime('%d %b %H:%M')
        print(f"[launch] lawn map : {_c.get('calibration_type','?')}  "
              f"{_c.get('lawn_width_m')} x {_c.get('lawn_length_m')} m  "
              f"(measured {_when})")
        if _c.get('rest_pose'):
            print('          rest pose recorded — the arm will return to it')
        else:
            print('          NO REST POSE in this calibration.')
            print('          The arm will NOT be moved, so the lawn plane is only')
            print('          right if the arm happens to be exactly where this was')
            print('          measured. To fix it, recalibrate — it records the pose:')
            print('              python lawn_sentry/calibrate/calibrate_lawn_scan.py')
        if _age_h > 6 and not _c.get('rest_pose'):
            print(f"          ^ that is {_age_h:.0f} hours old. If the camera has been")
            print("            moved or remounted since, the map WILL be wrong ->")
            print("            python lawn_sentry/calibrate/calibrate_lawn_scan.py")
    except Exception:
        pass

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


def _go_to_rest():
    """Put the arm where the calibration was measured, then let the port go.

    This happens BEFORE the web app and the scan start, for two reasons: the
    dashboard grabs the serial port as soon as the browser polls it, and the
    scan cannot afford to sit waiting without draining the camera queue.
    Doing it here means one owner at a time and no race.
    """
    import json as _json
    try:
        cal = _json.loads((DATA / 'lawn_calibration.json').read_text())
    except (OSError, ValueError):
        return
    rest = cal.get('rest_pose')
    if not rest:
        print('[launch] no rest pose in the calibration - the map will only be')
        print('         right if the arm is already where you calibrated it.')
        return
    try:
        from trimmer_arm import TrimmerArm
        arm = TrimmerArm()
    except Exception as e:
        print(f'[launch] arm not available ({e}) - cannot return to rest')
        return
    try:
        pose = {int(k): int(v) for k, v in rest.items()}
        print(f'[launch] moving to the calibrated rest pose {pose} ...')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(400, 20)
        arm.move_pose(pose)
        t0 = time.time()
        while time.time() - t0 < 8.0:
            if arm.arrived(pose, tol=25):
                break
            time.sleep(0.15)
        now = arm.read_pose()
        off = max(abs(now.get(k, v) - v) for k, v in pose.items()) if now else None
        if off is not None and off > 40:
            print(f'[launch] WARNING: arm stopped {off} ticks from the rest pose.')
            print('         Something may be obstructing it - the map will be off.')
        else:
            print('[launch] at rest pose - the lawn plane matches this view.')
    except Exception as e:
        print(f'[launch] could not reach the rest pose: {e}')
    finally:
        try:
            arm.close()          # release the port for the dashboard
        except Exception:
            pass


def main():
    if not _preflight():
        return
    _go_to_rest()
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
    except Exception:
        # Print it HERE, not on the way out. The finally block below blocks on
        # input(), so a crash in the scan used to sit invisible behind that
        # prompt: the symptom was "the video feed never appeared" with a
        # perfectly clean-looking log, and the traceback only surfaced after
        # pressing ENTER. Say it immediately and say what it means.
        import traceback
        print('\n' + '!' * 68)
        print('[launch] THE SCAN CRASHED — this is why there is no video feed.')
        print('!' * 68)
        traceback.print_exc()
        print('!' * 68 + '\n')
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
