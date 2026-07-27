# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import base64
import json
import re as _re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from paths import ROOT, SESSIONS, ensure as _ensure
_ensure()
HERE = ROOT
ACT_PORT = 8799            # tiny local endpoint the map's buttons talk to

# Bumped whenever the dashboard gains a feature the launcher must not silently
# skip. launch.py reuses an already-running web app; without this, a Streamlit
# process started before a feature existed keeps serving the OLD page forever
# and no amount of fixing the code changes what you see in the browser.
APP_BUILD = '2026.07.27-live-stream'

# ---- action endpoint (lets the in-map buttons persist changes) -------------
# The map runs in a sandboxed iframe with no way to write files, so its
# "Mark cleared" / "Not target" buttons call this localhost endpoint, which
# updates cleaned.json / dismissed.json in the session folder.
_FLOCK = threading.Lock()

# Resolving "which session is newest" means globbing + stat-ing the whole tree.
# The stream checks for a new frame every few milliseconds, so cache it briefly
# rather than hammering the filesystem.
_NEWEST = {'t': 0.0, 'dir': None}


def _newest_session(max_age=2.0):
    now = time.time()
    if now - _NEWEST['t'] < max_age:
        return _NEWEST['dir']
    d = None
    try:
        cands = [p for p in SESSIONS.glob('session_*')
                 if (p / 'session.json').is_file()]
        if cands:
            d = max(cands, key=lambda p: p.stat().st_mtime)
    except OSError:
        d = None
    _NEWEST.update(t=now, dir=d)
    return d


def _newest_live(max_age=2.0):
    """Path of the freshest live.jpg, or None."""
    d = _newest_session(max_age)
    if d is not None and (d / 'live.jpg').is_file():
        return d / 'live.jpg'
    try:
        c = [p / 'live.jpg' for p in SESSIONS.glob('session_*')
             if (p / 'live.jpg').is_file()]
        return max(c, key=lambda p: p.stat().st_mtime) if c else None
    except OSError:
        return None


class _ActHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):            # keep the console quiet
        pass

    def _head(self, code, ctype, length=None, extra=()):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store, max-age=0')
        self.send_header('Access-Control-Allow-Origin', '*')
        if length is not None:
            self.send_header('Content-Length', str(length))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()

    def _bytes(self, body, ctype):
        self._head(200, ctype, len(body))
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _404(self):
        self._head(404, 'text/plain', 0)

    # -- video ---------------------------------------------------------------
    def _serve_stream(self):
        """Push frames down ONE connection (multipart/x-mixed-replace).

        Far smoother than polling for individual JPEGs: no per-frame request,
        no connection setup, and the browser paints each part as it lands. We
        only write when the file's mtime changes, so an idle scan costs nothing.
        """
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store, max-age=0')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Connection', 'close')
        self.end_headers()
        last_m = 0.0
        starved = 0.0
        try:
            while True:
                p = _newest_live()
                m = 0.0
                if p is not None:
                    try:
                        m = p.stat().st_mtime
                    except OSError:
                        m = 0.0
                if m <= last_m:
                    time.sleep(0.008)
                    starved += 0.008
                    # nothing publishing for 20 s -> drop the connection so the
                    # client falls back and we don't leak a thread
                    if starved > 20.0:
                        break
                    continue
                try:
                    buf = p.read_bytes()
                except OSError:
                    continue
                if not buf:
                    continue
                last_m, starved = m, 0.0
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                 b'Content-Length: ' + str(len(buf)).encode()
                                 + b'\r\n\r\n' + buf + b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass
        self.close_connection = True

    def _serve_live(self):
        """Single newest frame — fallback for clients that can't stream."""
        p = _newest_live()
        try:
            img = p.read_bytes() if p is not None else None
        except OSError:
            img = None
        if not img:
            return self._404()
        self._bytes(img, 'image/jpeg')

    # -- data ----------------------------------------------------------------
    def _serve_data(self):
        """Current session state, so the dashboard can refresh itself.

        This is what lets the map update without Streamlit re-running the
        script and rebuilding the whole iframe (which used to restart the
        video every 2.5 s).
        """
        d = _newest_session()
        out = {}
        if d is not None:
            def rd(name, dflt=None):
                try:
                    return json.loads((d / name).read_text())
                except (OSError, ValueError):
                    return dflt
            s = rd('session.json') or {}
            out = {
                'targets': s.get('targets', []),
                'status': s.get('status', 'live'),
                'cam': s.get('camera'),
                'poly': s.get('lawn_polygon'),
                'polyArea': s.get('lawn_area_m2'),
                'W': s.get('lawn_width_m'), 'L': s.get('lawn_length_m'),
                'sdir': d.name,
                'session': d.name.replace('session_', ''),
                'cleaned': rd('cleaned.json', {}) or {},
                'dismissed': rd('dismissed.json', {}) or {},
            }
        self._bytes(json.dumps(out).encode(), 'application/json')

    def _serve_snap(self, q):
        """A target's capture photo, so it needn't be base64'd into the page."""
        sess = q.get('session', [''])[0]
        tid = q.get('id', [''])[0]
        if not _re.fullmatch(r'session_[0-9_]+', sess) or not tid.isdigit():
            return self._404()
        p = SESSIONS / sess / 'snapshots' / f'target_{int(tid):03d}.jpg'
        try:
            body = p.read_bytes()
        except OSError:
            return self._404()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Cache-Control', 'public, max-age=300')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if path.startswith('/version'):
            return self._bytes(json.dumps({'build': APP_BUILD}).encode(),
                               'application/json')
        if path.startswith('/stream'):
            return self._serve_stream()
        if path.startswith('/live'):
            return self._serve_live()
        if path.startswith('/data'):
            return self._serve_data()
        if path.startswith('/snap'):
            return self._serve_snap(q)
        sess = q.get('session', [''])[0]
        tid = q.get('id', [''])[0]
        action = q.get('action', [''])[0]
        ok = False
        if _re.fullmatch(r'session_[0-9_]+', sess) and tid.isdigit() \
                and action in ('clean', 'unclean', 'dismiss', 'restore'):
            fname = 'cleaned.json' if action in ('clean', 'unclean') else 'dismissed.json'
            fpath = SESSIONS / sess / fname
            with _FLOCK:
                try:
                    d = json.loads(fpath.read_text()) if fpath.exists() else {}
                    d[tid] = action in ('clean', 'dismiss')
                    fpath.write_text(json.dumps(d))
                    ok = True
                except OSError:
                    ok = False
        body = json.dumps({'ok': ok}).encode()
        self._head(200 if ok else 400, 'application/json', len(body))
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class _ActServer(ThreadingHTTPServer):
    daemon_threads = True             # streams must never block interpreter exit
    allow_reuse_address = True


def _start_act_server():
    try:
        srv = _ActServer(('127.0.0.1', ACT_PORT), _ActHandler)
    except OSError:
        return                            # already running from a previous rerun
    threading.Thread(target=srv.serve_forever, daemon=True).start()


_start_act_server()

st.set_page_config(page_title='RYOBI Lawn Sentry', page_icon='🟢',
                   layout='wide', initial_sidebar_state='collapsed')

# Own the whole page: hide Streamlit chrome, match the app background so the
# component sits seamlessly with no white gutter.
st.markdown("""<style>
  #MainMenu, footer, header {visibility: hidden;}
  .stApp {background: #0d0f0d;}
  .block-container {padding: 0.6rem 1rem 1.4rem; max-width: 1180px;}
  section[data-testid="stSidebar"] {background:#111311;}
</style>""", unsafe_allow_html=True)

# ---- live auto-refresh -----------------------------------------------------
# Deliberately SLOW. A Streamlit rerun rebuilds the component iframe from
# scratch, which tears down the video connection and re-runs every line of the
# dashboard's JS — the single biggest source of stutter. The component now
# refreshes itself from /data.json instead, so this only exists as a safety net
# for the case where the local endpoint is unreachable.
_HAVE_AUTOREFRESH = False
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key='live')
    _HAVE_AUTOREFRESH = True
except Exception:
    pass

# ---- session ---------------------------------------------------------------
# A scan run deletes old sessions while this app is auto-refreshing, so a
# folder can disappear (or be half-written) between listing it and reading it.
# Every read below is therefore defensive: never crash the dashboard because a
# file vanished — just fall back to the newest session that IS readable.
def list_sessions():
    if not SESSIONS.exists():
        return []
    return sorted([p for p in SESSIONS.glob('session_*')
                   if (p / 'session.json').is_file()], reverse=True)


def read_json(path, default=None):
    """Tolerant read: missing file, partial write or bad JSON -> default."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def waiting_screen(msg):
    st.markdown('## RYOBI Lawn Sentry')
    st.info(msg)
    st.caption('This page refreshes automatically.')
    time.sleep(2.0)
    st.rerun()


sessions = list_sessions()
if not sessions:
    waiting_screen('No scan session yet — run `python launch.py` to start scanning.')

with st.sidebar:
    st.markdown('### Session')
    names = [p.name for p in sessions]
    pick = st.selectbox('Scan session', names,
                        format_func=lambda n: n.replace('session_', ''),
                        label_visibility='collapsed')
    min_conf = st.slider('Minimum confidence', 0.0, 1.0, 0.0, 0.05)

chosen = SESSIONS / pick
data = read_json(chosen / 'session.json')
if data is None:
    # the chosen session was purged / is still being written — take the newest
    # readable one instead of crashing
    for cand in list_sessions():
        d = read_json(cand / 'session.json')
        if d is not None:
            chosen, data = cand, d
            break
if data is None:
    waiting_screen('Starting a new scan — waiting for the first data...')

targets = pd.DataFrame(data.get('targets', []))
if not targets.empty:
    targets = targets[targets.confidence >= min_conf].reset_index(drop=True)

# cleaned / dismissed persistence (per session)
cleaned_file = chosen / 'cleaned.json'
cleaned = read_json(cleaned_file, {}) or {}
dismissed_file = chosen / 'dismissed.json'
dismissed = read_json(dismissed_file, {}) or {}

W = float(data['lawn_width_m'])
L = float(data['lawn_length_m'])
all_records = [] if targets.empty else targets.to_dict('records')
records = [r for r in all_records if not dismissed.get(str(r['target_id']))]
dismissed_records = [r for r in all_records if dismissed.get(str(r['target_id']))]

# embed each target's snapshot as base64 so the map can show the real photo
# Capture photos are fetched from /snap.jpg on demand rather than base64'd into
# the page — with a dozen targets the old way pushed a megabyte through
# Streamlit on every rerun. A few of the newest are still embedded so the
# inspector shows something even if the endpoint is unreachable.
_EMBED_BUDGET = 260_000
_embedded = 0
for r in records:
    r['cleaned'] = bool(cleaned.get(str(r['target_id'])))
    r['photo'] = ''
for r in reversed(records):               # newest first
    snap = chosen / r.get('snapshot', '')
    try:
        if not snap.is_file():
            continue
        # base64 inflates by 4/3 — check the ENCODED size against the budget so
        # a couple of large captures can't blow past it
        if _embedded + snap.stat().st_size * 4 // 3 > _EMBED_BUDGET:
            continue
        r['photo'] = ('data:image/jpeg;base64,'
                      + base64.b64encode(snap.read_bytes()).decode())
        _embedded += len(r['photo'])
    except OSError:
        pass          # snapshot removed by a purge mid-refresh — skip the photo

# Embed the newest camera frame directly. The component ALSO polls the local
# endpoint for smooth video, but this guarantees a picture appears even if that
# request is blocked, and gives an instant first frame on load.
_live_b64 = ''
_live_age = None
try:
    _lj = chosen / 'live.jpg'
    if not _lj.is_file():
        _cands = sorted([p / 'live.jpg' for p in SESSIONS.glob('session_*')
                         if (p / 'live.jpg').is_file()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
        _lj = _cands[0] if _cands else None
    if _lj is not None and _lj.is_file():
        _live_age = round(time.time() - _lj.stat().st_mtime, 1)
        _b = _lj.read_bytes()
        if len(_b) < 1_500_000:
            _live_b64 = 'data:image/jpeg;base64,' + base64.b64encode(_b).decode()
except OSError:
    pass

PAYLOAD = json.dumps({
    'build': APP_BUILD,
    'liveInit': _live_b64, 'liveAge': _live_age,
    'targets': records, 'W': W, 'L': L,
    'status': data.get('status', 'live'),
    'session': chosen.name.replace('session_', ''),
    'sdir': chosen.name, 'actPort': ACT_PORT,
    'started': data.get('started', ''),
    # auto-detected lawn outline (metres). Absent -> the map draws a rectangle.
    'poly': data.get('lawn_polygon'),
    'polyArea': data.get('lawn_area_m2'),
    # live camera pose on the lawn (the lawn is fixed; the camera moves)
    'cam': data.get('camera'),
})

HTML = r"""
<div id="app"></div><div id="tip" class="tip"></div>
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:ital,wght@0,500;0,600;0,700;1,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --bg:#0d0f0d; --surf:#161915; --surf2:#1b1f19;
  --line:#282d25; --line2:#363c31; --txt:#eef1e9; --dim:#98a08f; --faint:#666e5f;
  --brand:#d4e021; --brand-dim:#9aa53f; --ok:#57c98a; --warn:#e0663a;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --disp:'Barlow Semi Condensed',system-ui,sans-serif;
  --ui:'Inter',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
#app{font-family:var(--ui); color:var(--txt); background:var(--surf);
     border:1px solid var(--line); border-radius:10px; overflow:hidden}
.hairline{height:1px;background:var(--line)}
/* ---- header ---- */
.hd{display:flex;align-items:center;gap:16px;padding:13px 18px;background:var(--bg)}
.mark{position:relative;display:flex;align-items:center;height:30px;padding:0 15px 0 12px;
      background:var(--brand);color:#111;font-family:var(--disp);font-weight:700;
      font-style:italic;font-size:20px;letter-spacing:.5px;
      clip-path:polygon(0 0,100% 0,90% 100%,0 100%)}
.hd .name{font-family:var(--disp);font-weight:700;font-size:16px;letter-spacing:2.5px;
          text-transform:uppercase;line-height:1}
.hd .name em{font-style:normal;color:var(--brand)}
.hd .sub{font-size:11px;color:var(--dim);letter-spacing:.3px;margin-top:3px}
.hd .sp{flex:1}
.conn{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;
      color:var(--dim);letter-spacing:.5px}
.conn .led{width:7px;height:7px;border-radius:50%;background:var(--brand);
           box-shadow:0 0 8px var(--brand);animation:led 2s ease-in-out infinite}
.conn.done .led{background:var(--dim);box-shadow:none;animation:none}
/* ---- kpi ---- */
.kpi{display:grid;grid-template-columns:repeat(4,1fr);background:var(--bg)}
.k{padding:12px 18px 14px;border-left:1px solid var(--line)}
.k:first-child{border-left:none}
.k .lab{font-size:9.5px;letter-spacing:1.6px;text-transform:uppercase;color:var(--faint);
        font-weight:600}
.k .val{font-family:var(--disp);font-weight:600;font-size:30px;line-height:1;margin-top:7px;
        font-variant-numeric:tabular-nums;letter-spacing:.5px}
.k .val.hot{color:var(--brand)}
.k .val .u{font-family:var(--ui);font-size:12px;font-weight:500;color:var(--dim);margin-left:3px}
.k .val.mono{font-family:var(--mono);font-size:19px;font-weight:500}
/* ---- body ---- */
.body{display:grid;grid-template-columns:1fr 322px;min-height:452px}
/* ---- live camera feed ---- */
.feed{position:relative;border:1px solid var(--line2);background:#0a0c0a;
      aspect-ratio:4/3;overflow:hidden;margin-bottom:14px}
.feed img{width:100%;height:100%;object-fit:cover;display:block}
.feed .off{position:absolute;inset:0;display:flex;align-items:center;
           justify-content:center;font-family:var(--mono);font-size:11px;
           color:var(--faint);text-align:center;padding:0 16px;line-height:1.7}
.feed .tag{position:absolute;left:0;top:0;display:flex;align-items:center;gap:6px;
           font-family:var(--mono);font-size:9.5px;letter-spacing:1.2px;
           color:#fff;background:rgba(10,12,10,.82);padding:4px 9px;
           border-bottom-right-radius:4px;z-index:2}
.feed .tag i{width:6px;height:6px;border-radius:50%;background:#e5484d;
             animation:led 1.6s ease-in-out infinite}
.feed .fps{position:absolute;right:0;bottom:0;font-family:var(--mono);
           font-size:9px;color:var(--dim);background:rgba(10,12,10,.8);
           padding:3px 7px;border-top-left-radius:4px;z-index:2}
.stage{position:relative;padding:14px 16px;border-right:1px solid var(--line);background:var(--surf)}
.tools{display:flex;gap:1px;margin-bottom:12px;background:var(--line);
       border:1px solid var(--line);border-radius:5px;width:max-content;overflow:hidden}
.tb{font-family:var(--ui);font-size:10.5px;font-weight:600;letter-spacing:1px;
    text-transform:uppercase;color:var(--dim);background:var(--surf2);
    padding:7px 13px;cursor:pointer;border:none;transition:.14s}
.tb:hover{color:var(--txt)}
.tb.on{background:var(--brand);color:#111}
.tb.gap{margin-left:auto}
svg.map{display:block;width:100%;height:auto;user-select:none;cursor:crosshair}
.coord{position:absolute;font-family:var(--mono);font-size:11px;color:var(--brand);
       background:rgba(13,15,13,.9);border:1px solid var(--line2);padding:3px 7px;
       border-radius:3px;pointer-events:none;opacity:0;transition:opacity .1s;white-space:nowrap}
/* reticle */
.rt{cursor:pointer}
.rt .hit{fill:transparent}
.rt .gauge{fill:none;stroke-linecap:round}
.rt .ring{fill:none;stroke:var(--line2)}
.rt .tick{stroke:var(--line2)}
.rt:hover .ring,.rt.sel .ring{stroke:var(--brand)}
.rt.sel .halo{animation:halo 2.2s ease-out infinite}
.rt.done{opacity:.4}
.rt .lead{stroke:var(--brand);stroke-dasharray:2 3;opacity:0}
.rt.sel .lead{opacity:1}
.chip{font-family:var(--mono);font-size:10px;fill:var(--txt)}
.rt .poo{font-size:14px;pointer-events:none}
.routeln{stroke:var(--brand);stroke-width:1.4;fill:none;stroke-dasharray:3 5;
         opacity:.85;animation:march 1s linear infinite}
/* ---- inspector ---- */
.insp{background:var(--surf2);padding:15px;display:flex;flex-direction:column;gap:0}
.eyebrow{font-size:9.5px;letter-spacing:1.7px;text-transform:uppercase;color:var(--faint);
         font-weight:600;margin-bottom:10px}
.shot{position:relative;border:1px solid var(--line2);background:#0a0c0a;aspect-ratio:4/3;
      overflow:hidden;margin-bottom:12px}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .none{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
            font-family:var(--mono);font-size:11px;color:var(--faint)}
.shot .cap{position:absolute;left:0;bottom:0;font-family:var(--mono);font-size:9.5px;
           letter-spacing:.5px;color:var(--brand);background:rgba(13,15,13,.85);
           padding:3px 7px;border-top-right-radius:4px}
.corner{position:absolute;width:11px;height:11px;border:1.5px solid var(--brand);opacity:.85}
.c-tl{top:6px;left:6px;border-right:0;border-bottom:0}
.c-tr{top:6px;right:6px;border-left:0;border-bottom:0}
.c-bl{bottom:6px;left:6px;border-right:0;border-top:0}
.c-br{bottom:6px;right:6px;border-left:0;border-top:0}
.tgt-id{font-family:var(--disp);font-weight:700;font-size:19px;letter-spacing:1px}
.tgt-id em{font-style:normal;color:var(--brand)}
.tag{font-family:var(--mono);font-size:9px;letter-spacing:.5px;padding:2px 6px;border-radius:3px;
     text-transform:uppercase;margin-left:8px;vertical-align:2px}
.tag.clr{color:var(--ok);border:1px solid var(--ok)}
.dl{margin-top:12px;border-top:1px solid var(--line);font-family:var(--mono)}
.dl .r{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--line);
       font-size:12px}
.dl .r .k2{color:var(--dim)} .dl .r .v2{color:var(--txt);font-weight:500}
.meter{display:flex;gap:2px;margin-top:12px}
.meter i{flex:1;height:5px;background:var(--line2);border-radius:1px}
.meter i.on{background:var(--brand)}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:15px}
.ab{font-family:var(--ui);font-size:10.5px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
    padding:9px 6px;border:1px solid var(--line2);background:transparent;color:var(--dim);
    cursor:pointer;border-radius:4px;transition:.14s}
.ab:hover{color:var(--txt);border-color:var(--dim)}
.ab.pri{background:var(--brand);border-color:var(--brand);color:#111}
.ab.pri:hover{background:#e4f02f}
.ab.dgr:hover{background:var(--warn);border-color:var(--warn);color:#fff}
.hint{font-size:12px;line-height:1.6;color:var(--dim);padding:2px 0 12px}
.list .row{display:flex;align-items:center;gap:10px;padding:9px 8px;margin:0 -8px;cursor:pointer;
           border-radius:5px;font-size:12px;border:1px solid transparent}
.list .row:hover{background:var(--surf)}
.list .row.sel{background:var(--surf);border-color:var(--line2)}
.list .row.done{opacity:.5}
.list .row .rid{font-family:var(--mono);font-size:11px;color:var(--brand);width:34px}
.list .row.done .rid{color:var(--dim)}
.list .row .rc{font-family:var(--mono);color:var(--dim);font-size:11px}
.list .row .rp{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--txt)}
.list .row .rx{opacity:0;color:var(--faint);font-size:14px;line-height:1;padding:0 2px}
.list .row:hover .rx{opacity:1}
.list .row .rx:hover{color:var(--warn)}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
       height:100%;gap:12px;text-align:center}
.empty .rlogo{width:46px;height:46px;border:1.5px solid var(--ok);border-radius:50%;
              display:flex;align-items:center;justify-content:center;color:var(--ok)}
.empty b{font-family:var(--disp);font-weight:600;font-size:18px;letter-spacing:1px}
.empty span{font-size:12px;color:var(--dim)}
/* ---- footer telemetry ---- */
.foot{display:flex;align-items:center;gap:22px;padding:9px 18px;background:var(--bg);
      border-top:1px solid var(--line);font-family:var(--mono);font-size:10.5px;
      color:var(--faint);letter-spacing:.4px}
.foot .b{color:var(--dim)}
.foot .sp{flex:1}
.foot .fled{width:6px;height:6px;border-radius:50%;background:var(--brand);
            display:inline-block;margin-right:6px;animation:led 2s ease-in-out infinite}
/* ---- banner ---- */
.banner{position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:9;
        display:flex;align-items:center;gap:9px;background:var(--brand);color:#111;
        font-family:var(--ui);font-size:11.5px;font-weight:700;letter-spacing:1.2px;
        text-transform:uppercase;padding:8px 15px;border-radius:4px;
        box-shadow:0 8px 26px rgba(212,224,33,.3);
        animation:bin .4s cubic-bezier(.2,1.3,.4,1) both, bout .5s ease 3.4s both}
/* a target that has gone from the lawn — neutral, not an alarm */
.banner.clr{background:var(--surf2);color:var(--txt);border:1px solid var(--line2);
            box-shadow:0 8px 26px rgba(0,0,0,.45)}
.tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:50;
     background:rgba(10,12,10,.96);border:1px solid var(--line2);border-radius:5px;
     padding:8px 11px;font-family:var(--mono);font-size:11px;color:var(--txt);
     box-shadow:0 10px 30px rgba(0,0,0,.5);white-space:nowrap}
.tip b{color:var(--brand)}
@keyframes led{0%,100%{opacity:1}50%{opacity:.35}}
@keyframes halo{0%{transform:scale(.7);opacity:.5}80%{opacity:0}100%{transform:scale(2);opacity:0}}
@keyframes march{to{stroke-dashoffset:-16}}
@keyframes bin{from{opacity:0;transform:translate(-50%,-14px)}to{opacity:1;transform:translate(-50%,0)}}
@keyframes bout{to{opacity:0;transform:translate(-50%,-8px)}}
@keyframes drop{0%{opacity:0;transform:translateY(-30px) scale(.5)}70%{opacity:1;transform:translateY(3px) scale(1.05)}100%{opacity:1;transform:translateY(0) scale(1)}}
.rt.enter{animation:drop .6s cubic-bezier(.2,1.25,.35,1) both;animation-delay:var(--d)}
</style>
<script>
const D = __PAYLOAD__;
const W=D.W, L=D.L, T=D.targets||[];
let sel=null, grid=false, route=false, prev=-1;
const SK='sentry2_'+D.session;
try{const s=JSON.parse(localStorage.getItem(SK)||'{}');
  if(typeof s.sel==='number'&&s.sel<T.length)sel=s.sel;
  if(typeof s.grid==='boolean')grid=s.grid;
  if(typeof s.route==='boolean')route=s.route;
  if(typeof s.count==='number')prev=s.count;}catch(e){}
function save(){try{localStorage.setItem(SK,JSON.stringify({sel,grid,route,count:T.length}));}catch(e){}}

// ---- helpers ----
function ago(iso){if(!iso)return'';const s=((Date.now()-new Date(iso).getTime())/1000)|0;
  if(s<0||isNaN(s))return'';if(s<60)return s+'s';if(s<3600)return(s/60|0)+'m '+(s%60)+'s';
  return(s/3600|0)+'h '+((s%3600)/60|0)+'m';}
function clock(iso){if(!iso)return'--:--:--';const d=new Date(iso);
  return isNaN(d)?'--:--:--':d.toTimeString().slice(0,8);}
function tier(c){return c>=0.8?'#d4e021':c>=0.6?'#c2cf5f':'#96a06a';}
function route_(){const open=T.map((t,i)=>({...t,i})).filter(t=>!t.cleaned);
  let cur={x_m:W/2,y_m:0},left=[...open],ord=[],dist=0;
  while(left.length){let bi=0,bd=1e9;left.forEach((t,k)=>{const d=Math.hypot(t.x_m-cur.x_m,t.y_m-cur.y_m);if(d<bd){bd=d;bi=k;}});
    dist+=bd;cur=left[bi];ord.push(cur);left.splice(bi,1);}return{ord,dist};}
function chime(){try{const a=new(window.AudioContext||window.webkitAudioContext)();
  [[784,0],[1175,.13]].forEach(([f,t])=>{const o=a.createOscillator(),g=a.createGain();
  o.frequency.value=f;o.type='sine';o.connect(g);g.connect(a.destination);
  g.gain.setValueAtTime(.001,a.currentTime+t);g.gain.exponentialRampToValueAtTime(.1,a.currentTime+t+.02);
  g.gain.exponentialRampToValueAtTime(.001,a.currentTime+t+.22);o.start(a.currentTime+t);o.stop(a.currentTime+t+.28);});}catch(e){}}
function act(a,tid,i){try{fetch(`http://localhost:${D.actPort}/act?session=${D.sdir}&id=${tid}&action=${a}`).catch(()=>{});}catch(e){}
  if(a==='clean')T[i].cleaned=true;if(a==='unclean')T[i].cleaned=false;
  if(a==='dismiss'){T.splice(i,1);if(sel===i)sel=null;else if(sel>i)sel-=1;}save();render();}

// ---- geometry ----
// If the scan measured a real lawn outline, fit the view to THAT; otherwise
// fall back to the calibrated rectangle. Everything downstream (pins, route,
// coordinate readout) uses the same X()/Y() so nothing else changes.
const POLY = (D.poly && D.poly.length>=3) ? D.poly : null;
let MNX=0, MNY=0, MXX=W, MXY=L;
if(POLY){
  MNX=Math.min(...POLY.map(p=>p[0])); MXX=Math.max(...POLY.map(p=>p[0]));
  MNY=Math.min(...POLY.map(p=>p[1])); MXY=Math.max(...POLY.map(p=>p[1]));
  const padm=0.15;                       // small margin so the edge isn't flush
  MNX-=padm; MNY-=padm; MXX+=padm; MXY+=padm;
}
// If a camera pose is known, make sure it is inside the view even when the
// camera walks off the lawn edge.
const CAM = D.cam || null;
if(CAM){
  MNX=Math.min(MNX,CAM.x_m-0.4); MXX=Math.max(MXX,CAM.x_m+0.4);
  MNY=Math.min(MNY,CAM.y_m-0.4); MXY=Math.max(MXY,CAM.y_m+0.4);
}
const SPANX=Math.max(0.5,MXX-MNX), SPANY=Math.max(0.5,MXY-MNY);
// KEEP THE ASPECT RATIO TRUE: one metre must be the same number of pixels in
// x and y, otherwise a square lawn looks stretched and distances lie.
const VW=660, PAD=34;
const IW=VW-2*PAD;
const Sk=IW/SPANX;                       // pixels per metre (uniform)
const VH=Math.round(SPANY*Sk)+2*PAD;
const IH=VH-2*PAD;
const SXk=Sk, SYk=Sk;
const X=x=>PAD+(x-MNX)*Sk, Y=y=>VH-PAD-(y-MNY)*Sk;
// where to draw the sensor: its live pose if we have one, else the near edge
const camX=CAM?X(CAM.x_m):X(POLY?(MNX+MXX)/2:W/2), camY=CAM?Y(CAM.y_m):Y(0);
const camHeading=CAM?CAM.heading_deg:90;
function polar(cx,cy,r,deg){const a=(deg-90)*Math.PI/180;return[cx+r*Math.cos(a),cy+r*Math.sin(a)];}
function arcPath(cx,cy,r,a0,a1){const[sx,sy]=polar(cx,cy,r,a1),[ex,ey]=polar(cx,cy,r,a0);
  const large=(a1-a0)<=180?0:1;return`M ${sx.toFixed(2)} ${sy.toFixed(2)} A ${r} ${r} 0 ${large} 0 ${ex.toFixed(2)} ${ey.toFixed(2)}`;}

// ---- svg pieces ----
function turf(){let s='';const n=Math.max(5,Math.round(L*2));
  for(let i=0;i<n;i++){const y0=PAD+i*(IH/n);s+=`<rect x="${PAD}" y="${y0.toFixed(1)}" width="${IW}" height="${(IH/n).toFixed(1)}" fill="${i%2?'#20281c':'#232c1f'}"/>`;}
  return s;}
function frameTicks(){let s='';
  for(let m=0;m<=W+1e-6;m+=0.5){const x=X(m),maj=Math.abs(m-Math.round(m))<1e-6;
    s+=`<line x1="${x.toFixed(1)}" y1="${VH-PAD}" x2="${x.toFixed(1)}" y2="${VH-PAD+(maj?7:4)}" stroke="#3a4133" stroke-width="1"/>`;
    if(maj&&m<=W)s+=`<text x="${x.toFixed(1)}" y="${VH-PAD+18}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="#666e5f">${m.toFixed(0)}</text>`;}
  for(let m=0;m<=L+1e-6;m+=0.5){const y=Y(m),maj=Math.abs(m-Math.round(m))<1e-6;
    s+=`<line x1="${PAD}" y1="${y.toFixed(1)}" x2="${PAD-(maj?7:4)}" y2="${y.toFixed(1)}" stroke="#3a4133" stroke-width="1"/>`;
    if(maj&&m<=L&&m>0)s+=`<text x="${PAD-11}" y="${(y+3).toFixed(1)}" text-anchor="end" font-family="IBM Plex Mono" font-size="9" fill="#666e5f">${m.toFixed(0)}</text>`;}
  return s;}
function gridLines(){if(!grid)return'';let s='';
  for(let m=0.5;m<W;m+=0.5)s+=`<line x1="${X(m).toFixed(1)}" y1="${PAD}" x2="${X(m).toFixed(1)}" y2="${VH-PAD}" stroke="#ffffff" stroke-opacity=".05" stroke-width="1"/>`;
  for(let m=0.5;m<L;m+=0.5)s+=`<line x1="${PAD}" y1="${Y(m).toFixed(1)}" x2="${VW-PAD}" y2="${Y(m).toFixed(1)}" stroke="#ffffff" stroke-opacity=".05" stroke-width="1"/>`;
  return s;}
function scaleBar(){const x0=PAD,x1=X(1),y=VH-12;
  return`<line x1="${x0}" y1="${y}" x2="${x1.toFixed(1)}" y2="${y}" stroke="#98a08f" stroke-width="1.5"/>
    <line x1="${x0}" y1="${y-3}" x2="${x0}" y2="${y+3}" stroke="#98a08f" stroke-width="1.5"/>
    <line x1="${x1.toFixed(1)}" y1="${y-3}" x2="${x1.toFixed(1)}" y2="${y+3}" stroke="#98a08f" stroke-width="1.5"/>
    <text x="${((x0+x1)/2).toFixed(1)}" y="${y-6}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="#98a08f">1 m</text>`;}
function sensor(){
  // The camera is drawn AT ITS LIVE POSE and its field of view sweeps round
  // with it, so walking to another side of the lawn moves the icon rather
  // than rotating the lawn.
  const fovDeg=69, reach=Math.max(SPANX,SPANY)*1.1;
  const a0=(camHeading-fovDeg/2)*Math.PI/180, a1=(camHeading+fovDeg/2)*Math.PI/180;
  const p1=[camX+Math.cos(a0)*reach*Sk, camY-Math.sin(a0)*reach*Sk];
  const p2=[camX+Math.cos(a1)*reach*Sk, camY-Math.sin(a1)*reach*Sk];
  const cone=`${camX},${camY} ${p1[0].toFixed(1)},${p1[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  const rot=-camHeading+90;      // svg y is down, icon drawn pointing "up"
  return`<polygon points="${cone}" fill="#d4e021" fill-opacity=".05"
            stroke="#d4e021" stroke-opacity=".2" stroke-width="1" stroke-dasharray="4 5"/>
    <g transform="translate(${camX},${camY}) rotate(${rot.toFixed(1)})">
      <rect x="-16" y="-7" width="32" height="15" rx="2.5" fill="#0d0f0d" stroke="#d4e021" stroke-width="1.2"/>
      <circle cx="-6" cy="0.5" r="3.5" fill="#111" stroke="#d4e021" stroke-width="1.3"/>
      <circle cx="6" cy="0.5" r="3.5" fill="#111" stroke="#4a5140" stroke-width="1.3"/>
      <path d="M 0 -9 L -4 -14 L 4 -14 Z" fill="#d4e021"/>
    </g>
    <text x="${camX}" y="${camY+26}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" letter-spacing="1" fill="#98a08f">OAK-D${CAM?' '+camHeading.toFixed(0)+'°':''}</text>`;}
function reticle(t,i){const cx=Math.min(Math.max(t.x_m,0),W),cy=Math.min(Math.max(t.y_m,0),L);
  const mx=X(cx),my=Y(cy),col=tier(t.confidence),selc=(i===sel);
  const g0=-118,g1=g0+t.confidence*236,gp=arcPath(0,0,15,g0,g1);
  let lead='';
  if(selc){const up=my>70;const ly=up?-30:30;
    lead=`<line class="lead" x1="0" y1="0" x2="0" y2="${ly}"/>
      <g transform="translate(0,${ly})"><rect x="-30" y="${up?-15:-2}" width="60" height="17" rx="2" fill="#0d0f0d" stroke="${col}" stroke-width="1"/>
      <text class="chip" x="0" y="${up?-3:11}" text-anchor="middle">${t.x_m.toFixed(2)},${t.y_m.toFixed(2)}</text></g>`;}
  const core=t.cleaned
    ?`<path d="M -4 0 L -1 3 L 5 -4" stroke="${col}" stroke-width="1.8" fill="none" stroke-linecap="round"/>`
    :`<text class="poo" text-anchor="middle" y="5">💩</text>`;
  return`<g class="rt ${selc?'sel':''} ${t.cleaned?'done':''} ${prev>=0&&i>=prev?'enter':''}"
       style="--d:${(i*.08).toFixed(2)}s" data-i="${i}" transform="translate(${mx.toFixed(1)},${my.toFixed(1)})">
    <circle class="halo" r="15" fill="none" stroke="${col}" stroke-width="1.2" opacity="0"/>
    ${lead}
    <circle class="hit" r="20"/>
    ${t.cleaned?'':`<path class="gauge" d="${gp}" stroke="${col}" stroke-width="2.4" opacity=".92"/>`}
    <circle class="ring" r="10.5" stroke-width="1.3"/>
    <line class="tick" x1="0" y1="-13" x2="0" y2="-16.5" stroke-width="1.2"/>
    <line class="tick" x1="0" y1="13" x2="0" y2="16.5" stroke-width="1.2"/>
    <line class="tick" x1="-13" y1="0" x2="-16.5" y2="0" stroke-width="1.2"/>
    <line class="tick" x1="13" y1="0" x2="16.5" y2="0" stroke-width="1.2"/>
    ${core}
  </g>`;}
function routePath(){if(!route)return'';const{ord,dist}=route_();if(!ord.length)return'';
  let pts=`${camX.toFixed(1)},${camY.toFixed(1)}`;ord.forEach(t=>{pts+=` ${X(t.x_m).toFixed(1)},${Y(t.y_m).toFixed(1)}`;});
  const last=ord[ord.length-1];
  return`<polyline class="routeln" points="${pts}"/>`+
    ord.map((t,k)=>`<text x="${(X(t.x_m)+16).toFixed(1)}" y="${(Y(t.y_m)-11).toFixed(1)}" font-family="IBM Plex Mono" font-size="9.5" font-weight="500" fill="#d4e021">${k+1}</text>`).join('')+
    `<text x="${X(last.x_m).toFixed(1)}" y="${(Y(last.y_m)+26).toFixed(1)}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9.5" fill="#98a08f">ROUTE ${dist.toFixed(1)} m</text>`;}
function polyPoints(){return POLY.map(p=>`${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');}
function mapSvg(){
  // custom outline mode: clip the turf to the real measured lawn shape
  const shapeDefs = POLY
    ? `<clipPath id="lawnclip"><polygon points="${polyPoints()}"/></clipPath>` : '';
  const shapeEdge = POLY
    ? `<polygon points="${polyPoints()}" fill="none" stroke="#0d0f0d" stroke-width="3"/>
       <polygon points="${polyPoints()}" fill="none" stroke="#3a4133" stroke-width="1.2"/>`
    : `<rect x="${PAD}" y="${PAD}" width="${IW}" height="${IH}" fill="none" stroke="#0d0f0d" stroke-width="2"/>
       <rect x="${PAD}" y="${PAD}" width="${IW}" height="${IH}" fill="none" stroke="#3a4133" stroke-width="1"/>`;
  return`<svg class="map" viewBox="0 0 ${VW} ${VH}">
    <defs>${shapeDefs}</defs>
    ${POLY?'':`<rect x="${PAD-5}" y="${PAD-5}" width="${IW+10}" height="${IH+10}" rx="6" fill="#1a2016"/>`}
    <g ${POLY?'clip-path="url(#lawnclip)"':''}>${turf()}</g>${gridLines()}
    ${shapeEdge}
    ${frameTicks()}${scaleBar()}
    <g id="cross" opacity="0"><line id="cx" stroke="#d4e021" stroke-width=".7" stroke-dasharray="2 3"/><line id="cy" stroke="#d4e021" stroke-width=".7" stroke-dasharray="2 3"/></g>
    ${sensor()}${routePath()}
    ${sel!==null&&T[sel]?`<line x1="${camX}" y1="${camY}" x2="${X(T[sel].x_m).toFixed(1)}" y2="${Y(T[sel].y_m).toFixed(1)}" stroke="#d4e021" stroke-width="1" stroke-dasharray="5 4" opacity=".55"/>`:''}
    <text x="${PAD}" y="18" font-family="IBM Plex Mono" font-size="9" letter-spacing="1" fill="#666e5f">PLAN VIEW · LAWN ${W.toFixed(1)}×${L.toFixed(1)} m${CAM?' · SENSOR '+CAM.x_m.toFixed(2)+','+CAM.y_m.toFixed(2)+' m':''}</text>
    ${T.map((t,i)=>reticle(t,i)).join('')}
  </svg>`;}

// ---- inspector ----
// Capture photo: fetch from the local endpoint (cheap, cached by the browser)
// and fall back to the embedded copy only if that request fails.
function shot(t){
  const url=`http://localhost:${D.actPort}/snap.jpg?session=${encodeURIComponent(D.sdir)}&id=${t.target_id}`;
  const fb=t.photo||'';
  if(fb) return `<img src="${url}" onerror="this.onerror=null;this.src='${fb}'">`;
  return `<img src="${url}" onerror="this.style.display='none';this.insertAdjacentHTML('afterend','&lt;div class=\\'none\\'&gt;NO CAPTURE&lt;/div&gt;')">`;
}
function insp(){const open=T.filter(t=>!t.cleaned).length;
  if(!T.length)return`<div class="empty"><div class="rlogo">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M5 13l4 4L19 7"/></svg>
    </div><b>AREA CLEAR</b><span>No targets detected in this sweep</span></div>`;
  let h='';
  if(sel!==null&&T[sel]){const t=T[sel],on=Math.round(t.confidence*10);

    h+=`<div class="eyebrow">Target inspector</div>
    <div class="shot">
      <span class="corner c-tl"></span><span class="corner c-tr"></span><span class="corner c-bl"></span><span class="corner c-br"></span>
      ${shot(t)}
      <span class="cap">SENSOR CAPTURE · TGT-${String(t.target_id).padStart(2,'0')}</span>
    </div>
    <div><span class="tgt-id">TARGET <em>#${String(t.target_id).padStart(2,'0')}</em></span>${t.cleaned?'<span class="tag clr">Cleared</span>':''}</div>
    <div class="dl">
      <div class="r"><span class="k2">POS X</span><span class="v2">${t.x_m.toFixed(2)} m</span></div>
      <div class="r"><span class="k2">POS Y</span><span class="v2">${t.y_m.toFixed(2)} m</span></div>
      <div class="r"><span class="k2">RANGE</span><span class="v2">${Math.hypot(t.x_m-W/2,t.y_m).toFixed(2)} m</span></div>
      <div class="r"><span class="k2">FIRST SEEN</span><span class="v2">${clock(t.timestamp)}</span></div>
      <div class="r"><span class="k2">OBSERVED</span><span class="v2">${t.observations}×</span></div>
      <div class="r"><span class="k2">CONFIDENCE</span><span class="v2">${(t.confidence*100).toFixed(0)}%</span></div>
    </div>
    <div class="meter">${Array.from({length:10},(_,k)=>`<i class="${k<on?'on':''}"></i>`).join('')}</div>
    <div class="acts">
      ${t.cleaned?`<button class="ab" data-act="unclean" data-tid="${t.target_id}" data-i="${sel}">Re-open</button>`
                :`<button class="ab pri" data-act="clean" data-tid="${t.target_id}" data-i="${sel}">Mark cleared</button>`}
      <button class="ab dgr" data-act="dismiss" data-tid="${t.target_id}" data-i="${sel}">Not target</button>
    </div>`;
  }else{
    h+=`<div class="eyebrow">Detections · ${open} active</div>
    <div class="hint">Select a target on the plan to inspect its capture, coordinates and range from the unit.</div>
    <div class="list">`+T.map((t,i)=>`<div class="row ${t.cleaned?'done':''}" data-i="${i}">
        <span class="rid">${String(t.target_id).padStart(2,'0')}</span>
        <span class="rc">${t.x_m.toFixed(2)}, ${t.y_m.toFixed(2)}</span>
        <span class="rp">${(t.confidence*100).toFixed(0)}%</span>
        <span class="rx" data-act="dismiss" data-tid="${t.target_id}" data-i="${i}" title="Not a target">×</span></div>`).join('')+`</div>`;
  }
  return h;}

// ---- render ----
function render(){const open=T.filter(t=>!t.cleaned).length;
  const times=T.map(t=>t.timestamp).filter(Boolean).sort();
  const last=times.length?ago(times[times.length-1]):'--';
  const best=T.length?Math.max(...T.map(t=>(t.confidence*100)|0)):0;
  const done=D.status==='complete';
  document.getElementById('app').innerHTML=`
  <div class="hd">
    <div class="mark">RYOBI</div>
    <div><div class="name">LAWN <em>SENTRY</em></div><div class="sub">Autonomous waste detection system</div></div>
    <div class="sp"></div>
    <div class="conn ${done?'done':''}"><span class="led"></span>${done?'SWEEP COMPLETE':'UNIT ONLINE'}</div>
  </div>
  <div class="hairline"></div>
  <div class="kpi">
    <div class="k"><div class="lab">Active targets</div><div class="val hot">${String(open).padStart(2,'0')}</div></div>
    <div class="k"><div class="lab">Total detected</div><div class="val">${String(T.length).padStart(2,'0')}</div></div>
    <div class="k"><div class="lab">Area monitored</div><div class="val">${(D.polyArea!=null?D.polyArea:(W*L)).toFixed(1)}<span class="u">m²</span></div></div>
    <div class="k"><div class="lab">Last detection</div><div class="val mono">${last}</div></div>
  </div>
  <div class="hairline"></div>
  <div class="body">
    <div class="stage">
      <div class="tools">
        <button class="tb ${grid?'on':''}" id="bg">Grid</button>
        <button class="tb ${route?'on':''}" id="br">Route</button>
        <button class="tb" id="bc" ${sel===null?'style="opacity:.4;pointer-events:none"':''}>Clear</button>
      </div>
      ${mapSvg()}
      <div class="coord" id="rd"></div>
      <div id="alert"></div>
    </div>
    <div class="insp">
      <div class="eyebrow">Sensor feed</div>
      <div id="feedslot"></div>
      ${insp()}
    </div>
  </div>
  <div class="foot">
    <span><span class="fled"></span>SESSION <span class="b">${D.session}</span></span>
    <span>SENSOR <span class="b">OAK-D Lite · stereo</span></span>
    <span class="sp"></span>
    <span>ORIGIN near-left · +X right · +Y forward</span>
    <span title="dashboard build — if this looks old, the web app needs restarting">${D.build||'?'}</span>
  </div>`;
  // wire actions FIRST (stopPropagation) then selection
  document.querySelectorAll('[data-act]').forEach(el=>el.addEventListener('click',e=>{
    e.stopPropagation();act(el.dataset.act,el.dataset.tid,+el.dataset.i);}));
  document.querySelectorAll('.rt,.list .row').forEach(el=>el.addEventListener('click',()=>{
    sel=(+el.dataset.i===sel)?null:+el.dataset.i;save();render();}));
  const g=document.getElementById('bg');if(g)g.onclick=()=>{grid=!grid;save();render();};
  const r=document.getElementById('br');if(r)r.onclick=()=>{route=!route;save();render();};
  const c=document.getElementById('bc');if(c)c.onclick=()=>{sel=null;save();render();};
  // Re-adopt the SAME feed element rather than building a new one. Moving a
  // node keeps its in-flight image stream alive; re-creating it would restart
  // the video on every re-render.
  const slot=document.getElementById('feedslot');
  if(slot) slot.replaceWith(feedEl());
  bindCoord();
  startFeed();
}

// ---- live camera feed -------------------------------------------------------
// Poll the local endpoint for the newest annotated frame. Each image is loaded
// fully before the next request, so a slow frame throttles itself instead of
// queueing up. Runs independently of the page refresh, so the video stays
// smooth while the map updates.
// The panel is built ONCE and moved between renders, so the stream survives.
let feedNode=null;
function feedEl(){
  if(feedNode) return feedNode;
  feedNode=document.createElement('div');
  feedNode.className='feed';
  feedNode.innerHTML=`<span class="tag"><i></i>LIVE</span>`
    +`<img id="livefeed" alt="">`
    +`<div class="off" id="liveoff">connecting to the camera…</div>`
    +`<span class="fps" id="livefps"></span>`;
  return feedNode;
}

// Three ways to get pictures, best first:
//   1. MJPEG  — one connection, server pushes frames at camera rate. Smoothest.
//   2. poll   — one request per frame; works where streaming doesn't.
//   3. embed  — the frame Streamlit put in the page. Always works, slow.
let feedTimer=null, feedTimes=[], feedStarted=false, stallTimer=null;
function startFeed(){
  const img=document.getElementById('livefeed');
  const off=document.getElementById('liveoff');
  const fps=document.getElementById('livefps');
  if(!img) return;
  if(feedStarted) return;          // persistent node — never restart it
  feedStarted=true;

  const show=t=>{ if(off){ off.style.display='none'; } if(fps&&t!==undefined) fps.textContent=t; };
  const hide=t=>{ if(off){ off.style.display='flex'; off.textContent=t; } if(fps) fps.textContent=''; };

  if(D.liveInit){ img.src=D.liveInit; show('connecting'); }
  else hide('waiting for the camera…');

  function mark(){
    feedTimes.push(performance.now());
    while(feedTimes.length>16) feedTimes.shift();
    if(fps){
      if(feedTimes.length>4){
        const dt=(feedTimes[feedTimes.length-1]-feedTimes[0])/(feedTimes.length-1);
        fps.textContent=(1000/dt).toFixed(0)+' fps';
      }else fps.textContent='live';
    }
  }

  // --- 1. streaming -------------------------------------------------------
  function stream(){
    let got=false;
    const watchdog=setTimeout(()=>{ if(!got){ img.onload=img.onerror=null; poll(); } },3000);
    img.onload=()=>{                      // fires once per pushed frame
      got=true; clearTimeout(watchdog); show(); mark();
      clearTimeout(stallTimer);
      // if frames stop arriving the connection has dropped — reconnect
      stallTimer=setTimeout(()=>{ img.onload=img.onerror=null; feedTimes.length=0; stream(); },6000);
    };
    img.onerror=()=>{ clearTimeout(watchdog); clearTimeout(stallTimer);
                      img.onload=img.onerror=null; poll(); };
    img.src=`http://localhost:${D.actPort}/stream.mjpg?t=${Date.now()}`;
  }

  // --- 2. polling ---------------------------------------------------------
  let misses=0;
  function poll(){
    if(feedTimer) clearTimeout(feedTimer);
    const probe=new Image();
    probe.onload=()=>{
      img.src=probe.src; show(); mark(); misses=0;
      feedTimer=setTimeout(poll,70);
    };
    probe.onerror=()=>{
      misses++; feedTimes.length=0;
      // 3. keep the embedded frame on screen rather than going blank
      if(misses>2){ if(D.liveInit){ show('slow refresh'); }
                    else hide('no camera feed — start a scan with launch.py'); }
      feedTimer=setTimeout(poll, misses>4?1500:600);
    };
    probe.src=`http://localhost:${D.actPort}/live.jpg?t=${Date.now()}`;
  }

  stream();
}

// ---- self-refreshing data ---------------------------------------------------
// Pull session state straight from the local endpoint. This is what removes the
// lag: Streamlit no longer has to re-run and rebuild the iframe to show a new
// target, so the video is never interrupted and the map updates ~4x faster.
let dataTimer=null, sig=null;
function startData(){
  if(dataTimer) clearTimeout(dataTimer);
  const loop=()=>{
    fetch(`http://localhost:${D.actPort}/data.json?t=${Date.now()}`,{cache:'no-store'})
      .then(r=>r.ok?r.json():null)
      .then(j=>{ if(j&&j.sdir) applyData(j); dataTimer=setTimeout(loop,600); })
      .catch(()=>{ dataTimer=setTimeout(loop,3000); });   // endpoint gone: Streamlit still refreshes
  };
  loop();
}

function applyData(j){
  const dis=j.dismissed||{}, cl=j.cleaned||{};
  const list=(j.targets||[]).filter(t=>!dis[String(t.target_id)]);
  // carry the embedded photos across so nothing flickers
  const photos={}; T.forEach(t=>{ if(t.photo) photos[t.target_id]=t.photo; });
  list.forEach(t=>{ t.cleaned=!!cl[String(t.target_id)]; t.photo=photos[t.target_id]||''; });

  const s=JSON.stringify([list.map(t=>[t.target_id,+t.x_m.toFixed(3),+t.y_m.toFixed(3),
                                       t.cleaned,+(t.confidence||0).toFixed(3)]),
                          j.cam?[+j.cam.x_m.toFixed(2),+j.cam.y_m.toFixed(2),
                                 Math.round(j.cam.heading_deg)]:null,
                          j.status]);
  if(s===sig) return;                       // nothing changed — don't re-render
  const first=sig===null;
  const before=T.map(t=>t.target_id);
  sig=s;

  // Track the selection by id, not index: targets can now disappear.
  const selId=(sel!==null&&T[sel])?T[sel].target_id:null;
  D.sdir=j.sdir||D.sdir; D.session=j.session||D.session;
  D.cam=j.cam; D.status=j.status||D.status;
  if(j.polyArea!=null) D.polyArea=j.polyArea;
  T.length=0; list.forEach(t=>T.push(t));   // same array reference throughout
  sel=selId===null?null:T.findIndex(t=>t.target_id===selId);
  if(sel<0) sel=null;

  const added=T.filter(t=>!before.includes(t.target_id)).length;
  const gone=before.filter(id=>!T.some(t=>t.target_id===id)).length;
  render();
  if(!first&&added){
    chime(); sel=T.length-1; save(); render();
    const el=document.getElementById('alert');
    if(el) el.innerHTML=`<div class="banner">◈ ${added>1?added+' new targets':'New target'} detected</div>`;
  }else if(!first&&gone){
    save();
    const el=document.getElementById('alert');
    if(el){ el.innerHTML=`<div class="banner clr">◈ ${gone>1?gone+' targets':'Target'} cleared from lawn</div>`;
            setTimeout(()=>{ if(el.firstChild) el.innerHTML=''; },3400); }
  }else save();
}
// live coordinate readout under the cursor
function bindCoord(){const svg=document.querySelector('svg.map');if(!svg)return;
  const rd=document.getElementById('rd'),cross=document.getElementById('cross'),
        lx=document.getElementById('cx'),ly=document.getElementById('cy');
  svg.addEventListener('mousemove',e=>{const b=svg.getBoundingClientRect();
    const px=(e.clientX-b.left)/b.width*VW, py=(e.clientY-b.top)/b.height*VH;
    if(px<PAD||px>VW-PAD||py<PAD||py>VH-PAD){cross.setAttribute('opacity','0');rd.style.opacity=0;return;}
    const mx=(px-PAD)/SXk, my=(VH-PAD-py)/SYk;
    cross.setAttribute('opacity','1');
    lx.setAttribute('x1',px);lx.setAttribute('y1',PAD);lx.setAttribute('x2',px);lx.setAttribute('y2',VH-PAD);
    ly.setAttribute('x1',PAD);ly.setAttribute('y1',py);ly.setAttribute('x2',VW-PAD);ly.setAttribute('y2',py);
    rd.style.opacity=1;rd.textContent=`X ${mx.toFixed(2)}  Y ${my.toFixed(2)} m`;
    rd.style.left=(e.clientX-b.left+14)+'px';rd.style.top=(e.clientY-b.top+14)+'px';});
  svg.addEventListener('mouseleave',()=>{cross.setAttribute('opacity','0');rd.style.opacity=0;});}

render();
// New targets since the last visit (page was closed while scanning).
if(prev>=0&&T.length>prev){const n=T.length-prev;
  const put=()=>{const el=document.getElementById('alert');
    if(el)el.innerHTML=`<div class="banner">◈ ${n>1?n+' new targets':'New target'} detected</div>`;};
  chime();sel=T.length-1;save();render();put();}else{save();}
// From here the dashboard keeps itself current without any Streamlit rerun.
startData();
// keyboard: cycle targets
document.addEventListener('keydown',e=>{if(!T.length)return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){sel=(sel===null)?0:(sel+1)%T.length;save();render();}
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){sel=(sel===null)?T.length-1:(sel-1+T.length)%T.length;save();render();}
  else if(e.key==='Escape'){sel=null;save();render();}});
</script>
"""
components.html(HTML.replace('__PAYLOAD__', PAYLOAD), height=724, scrolling=False)

# Marking / dismissing happens on the plan itself. Streamlit keeps only a
# restore list for anything flagged "not target", plus the CSV export.
if dismissed_records:
    with st.expander(f'Flagged "not target" ({len(dismissed_records)}) — restore'):
        for r in dismissed_records:
            key = str(r['target_id'])
            c1, c2 = st.columns([3, 1])
            c1.write(f"Target #{r['target_id']} — {r['x_m']:.2f} m, {r['y_m']:.2f} m "
                     f"({r['confidence']:.0%})")
            if c2.button('Restore', key=f'rs_{key}', use_container_width=True):
                dismissed[key] = False
                dismissed_file.write_text(json.dumps(dismissed))
                st.rerun()

_csv = chosen / 'detections.csv'
if _csv.is_file():
    try:
        st.download_button('Download detection report (CSV)', _csv.read_bytes(),
                           'lawn_sentry_report.csv', 'text/csv')
    except OSError:
        pass

# ---- built-in live refresh (no extra package needed) -----------------------
if not _HAVE_AUTOREFRESH and data.get('status') != 'complete':
    import time as _t
    _t.sleep(2.5)
    st.rerun()
