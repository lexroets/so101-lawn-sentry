"""Why isn't the live video showing in the web app?

Run this WHILE launch.py is running (scan window open, browser open):

    python diag_feed.py

It walks the whole chain link by link and tells you which one is broken:

    scan_lawn writes live.jpg  ->  web app serves it  ->  browser displays it

Nothing here changes any settings; it only looks.
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
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

from paths import ROOT, SESSIONS, DASHBOARD
HERE = ROOT
APP = DASHBOARD / 'lawn_map_app.py'
ACT_PORT = 8799
WEB_PORT = 8501

OK, BAD, WARN = '  [ok]  ', '  [FAIL]', '  [warn]'
problems = []


def port_open(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def get(path, timeout=3.0):
    """-> (status, content_type, body) or (None, None, error string)"""
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{ACT_PORT}{path}')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get('Content-Type'), r.read()
    except urllib.error.HTTPError as e:
        return e.code, None, b''
    except Exception as e:
        return None, None, str(e).encode()


def section(n, title):
    print(f'\n{n}. {title}')
    print('   ' + '-' * (len(title) + 2))


print('=' * 66)
print('LIVE FEED DIAGNOSTIC')
print('=' * 66)

# ---------------------------------------------------------------- 1. frames
section(1, 'Is scan_lawn.py writing camera frames?')
live = None
if not SESSIONS.is_dir():
    print(BAD, 'no scan_sessions folder at all — the scan has never run')
    problems.append('Start a scan:  python launch.py')
else:
    cands = sorted([p for p in SESSIONS.glob('session_*')
                    if (p / 'live.jpg').is_file()],
                   key=lambda p: (p / 'live.jpg').stat().st_mtime, reverse=True)
    sess = sorted(SESSIONS.glob('session_*'))
    print(f'{OK} {len(sess)} session folder(s), {len(cands)} with a live.jpg')
    if not cands:
        print(BAD, 'NO live.jpg anywhere — scan_lawn.py is not publishing frames')
        print('        In the scan terminal you should see a line reading')
        print('        "live feed publishing -> session_.../live.jpg".')
        print('        If you see "live feed FAILED to write", the message says why.')
        problems.append('scan_lawn.py is not writing live.jpg — is the scan running?')
    else:
        live = cands[0] / 'live.jpg'
        age = time.time() - live.stat().st_mtime
        size = live.stat().st_size
        print(f'{OK} newest: {live.parent.name}/live.jpg  ({size/1024:.0f} KB)')
        if age > 5:
            print(WARN, f'but it is {age:.0f} s old — the scan is not running now')
            problems.append('live.jpg is stale — start the scan before checking')
        else:
            print(f'{OK} {age:.1f} s old — frames ARE being written right now')
        # is it actually growing? watch for a change
        m0 = live.stat().st_mtime
        time.sleep(1.2)
        m1 = live.stat().st_mtime if live.is_file() else m0
        if m1 > m0:
            print(f'{OK} file updated again during a 1.2 s watch — publishing live')
        elif age <= 5:
            print(WARN, 'file did not change in 1.2 s — scan may be paused')

# ------------------------------------------------------------- 2. web app up
section(2, 'Is the web app running, and is it the CURRENT build?')
if not port_open(WEB_PORT):
    print(BAD, f'nothing is listening on {WEB_PORT} — the web app is not running')
    problems.append('Start it:  python launch.py')
else:
    print(f'{OK} web app is up on {WEB_PORT}')

if not port_open(ACT_PORT):
    print(BAD, f'nothing on {ACT_PORT} — the data/video endpoint did not start')
    print('        This is what serves the video. Without it there can be no feed.')
    problems.append(f'Port {ACT_PORT} not listening — restart launch.py')
else:
    print(f'{OK} data/video endpoint is up on {ACT_PORT}')
    st, ct, body = get('/version')
    want = None
    try:
        import re
        m = re.search(r"^APP_BUILD\s*=\s*'([^']+)'", APP.read_text(), re.M)
        want = m.group(1) if m else None
    except OSError:
        pass
    if st != 200:
        print(BAD, 'the running web app has NO /version endpoint, so it was')
        print('        started from an older copy of lawn_map_app.py.')
        print('        THIS IS ALMOST CERTAINLY YOUR PROBLEM: the browser is')
        print('        showing an old dashboard that has no video panel.')
        problems.append('Stale web app. Fix:  lsof -ti tcp:8501 | xargs kill   '
                        'then rerun launch.py')
    else:
        have = json.loads(body).get('build')
        if want and have != want:
            print(BAD, f'running build {have!r} but the file on disk is {want!r}')
            print('        The web app is stale — restart it.')
            problems.append('Stale web app. Fix:  lsof -ti tcp:8501 | xargs kill   '
                            'then rerun launch.py')
        else:
            print(f'{OK} build {have} matches the code on disk')

# -------------------------------------------------------------- 3. endpoints
section(3, 'Can the endpoint actually hand over a picture?')
if port_open(ACT_PORT):
    st, ct, body = get('/live.jpg')
    if st == 200 and body:
        print(f'{OK} /live.jpg -> {st} {ct} {len(body)/1024:.0f} KB')
        if body[:2] != b'\xff\xd8':
            print(BAD, 'those bytes are not a JPEG')
            problems.append('live.jpg is corrupt')
    elif st == 404:
        print(BAD, '/live.jpg -> 404 (endpoint is up but finds no frame file)')
        problems.append('The web app cannot see live.jpg — is it running from '
                        'the same folder as scan_lawn.py?')
    else:
        print(BAD, f'/live.jpg -> {st} {body[:90]!r}')
        problems.append('The video endpoint is not responding properly')

    st, ct, body = get('/data.json')
    if st == 200:
        try:
            d = json.loads(body)
            print(f'{OK} /data.json -> {len(d.get("targets", []))} target(s), '
                  f'session {d.get("session", "?")}')
        except ValueError:
            print(BAD, '/data.json returned non-JSON')
    else:
        print(WARN, f'/data.json -> {st} (map self-refresh unavailable)')

    # streaming
    try:
        import http.client
        c = http.client.HTTPConnection('127.0.0.1', ACT_PORT, timeout=6)
        c.request('GET', '/stream.mjpg')
        r = c.getresponse()
        buf = b''
        t0 = time.time()
        while time.time() - t0 < 2.0:
            chunk = r.read(8192)
            if not chunk:
                break
            buf += chunk
        c.close()
        n = buf.count(b'--frame')
        if r.status == 200 and n:
            print(f'{OK} /stream.mjpg -> {n} frames in 2 s ({n/2:.0f} fps)')
        elif r.status == 200:
            print(WARN, '/stream.mjpg opened but pushed no frames '
                        '(nothing is publishing)')
        else:
            print(BAD, f'/stream.mjpg -> {r.status}')
            problems.append('Streaming endpoint missing — the web app is stale')
    except Exception as e:
        print(BAD, f'/stream.mjpg failed: {e}')

# ------------------------------------------------------------------ verdict
print('\n' + '=' * 66)
if not problems:
    print('VERDICT: every link in the chain is working.')
    print()
    print('If the panel is still blank in the browser, it is the browser end:')
    print('  1. Hard-reload the page:  Cmd-Shift-R')
    print('  2. Open the browser console (Cmd-Option-J) and look for red errors')
    print('  3. Open this URL directly in a new tab — you should see a photo:')
    print(f'         http://localhost:{ACT_PORT}/live.jpg')
    print('     If that works but the dashboard does not, tell me and I will')
    print('     route the video through the web app\'s own port instead.')
else:
    print('VERDICT: found %d problem(s).' % len(problems))
    for i, p in enumerate(problems, 1):
        print(f'  {i}. {p}')
print('=' * 66)
