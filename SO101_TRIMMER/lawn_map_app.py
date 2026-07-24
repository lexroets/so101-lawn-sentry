"""RYOBI Lawn Sentry — pet-waste detection dashboard.

Run:  streamlit run lawn_map_app.py   (or just python launch.py)

Design: RYOBI's lime-on-charcoal industrial language. The map is an
interactive canvas — click a detection pin to open its detail panel with the
real camera snapshot, a measuring line and distance from the mower; toggle the
half-metre grid; mark targets as cleaned (persisted per session).
"""
import base64
import json
import re as _re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / 'scan_sessions'
ACT_PORT = 8799            # tiny local endpoint the map's buttons talk to

# ---- action endpoint (lets the in-map buttons persist changes) -------------
# The map runs in a sandboxed iframe with no way to write files, so its
# "Mark cleaned" / "Not poop" buttons call this localhost endpoint, which
# updates cleaned.json / dismissed.json in the session folder.
_FLOCK = threading.Lock()


class _ActHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # keep the console quiet
        pass

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        sess = q.get('session', [''])[0]
        tid = q.get('id', [''])[0]
        action = q.get('action', [''])[0]
        ok = False
        if _re.fullmatch(r'session_[0-9_]+', sess) and tid.isdigit() \
                and action in ('clean', 'unclean', 'dismiss', 'restore'):
            fname = 'cleaned.json' if action in ('clean', 'unclean') else 'dismissed.json'
            path = SESSIONS / sess / fname
            with _FLOCK:
                try:
                    d = json.loads(path.read_text()) if path.exists() else {}
                    d[tid] = action in ('clean', 'dismiss')
                    path.write_text(json.dumps(d))
                    ok = True
                except OSError:
                    ok = False
        body = json.dumps({'ok': ok}).encode()
        self.send_response(200 if ok else 400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_act_server():
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', ACT_PORT), _ActHandler)
    except OSError:
        return                            # already running from a previous rerun
    threading.Thread(target=srv.serve_forever, daemon=True).start()


_start_act_server()

st.set_page_config(page_title='RYOBI Lawn Sentry', page_icon='🟢',
                   layout='wide', initial_sidebar_state='collapsed')

# Hide Streamlit chrome so the component owns the page.
st.markdown("""<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding: 0.8rem 1.2rem 2rem; max-width: 1200px;}
</style>""", unsafe_allow_html=True)

# ---- live auto-refresh -----------------------------------------------------
# Preferred: streamlit-autorefresh (non-blocking). Fallback: a built-in
# sleep+rerun at the END of the script, so live updates work with NO extra
# package installed. _HAVE_AUTOREFRESH gates the fallback.
_HAVE_AUTOREFRESH = False
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=2500, key='live')
    _HAVE_AUTOREFRESH = True
except Exception:
    pass

# ---- session ---------------------------------------------------------------
sessions = sorted([p for p in SESSIONS.glob('session_*')
                   if (p / 'session.json').exists()], reverse=True) \
    if SESSIONS.exists() else []
if not sessions:
    st.markdown('## RYOBI Lawn Sentry')
    st.info('No scan sessions yet — run `python launch.py` to start scanning.')
    st.stop()

with st.sidebar:
    st.markdown('### Session')
    chosen = st.selectbox('Scan session', sessions,
                          format_func=lambda p: p.name.replace('session_', ''),
                          label_visibility='collapsed')
    min_conf = st.slider('Minimum confidence', 0.0, 1.0, 0.0, 0.05)

data = json.loads((chosen / 'session.json').read_text())
targets = pd.DataFrame(data.get('targets', []))
if not targets.empty:
    targets = targets[targets.confidence >= min_conf].reset_index(drop=True)

# cleaned / dismissed persistence (per session)
cleaned_file = chosen / 'cleaned.json'
cleaned = json.loads(cleaned_file.read_text()) if cleaned_file.exists() else {}
dismissed_file = chosen / 'dismissed.json'
dismissed = json.loads(dismissed_file.read_text()) if dismissed_file.exists() else {}

W = float(data['lawn_width_m'])
L = float(data['lawn_length_m'])
all_records = [] if targets.empty else targets.to_dict('records')
# "not poop" targets are removed from the lawn entirely
records = [r for r in all_records if not dismissed.get(str(r['target_id']))]
dismissed_records = [r for r in all_records if dismissed.get(str(r['target_id']))]

# embed each target's snapshot as base64 so the map can show the real photo
for r in records:
    r['cleaned'] = bool(cleaned.get(str(r['target_id'])))
    r['photo'] = ''
    snap = chosen / r.get('snapshot', '')
    if snap.is_file() and snap.stat().st_size < 900_000:
        r['photo'] = ('data:image/jpeg;base64,'
                      + base64.b64encode(snap.read_bytes()).decode())

PAYLOAD = json.dumps({
    'targets': records, 'W': W, 'L': L,
    'status': data.get('status', 'live'),
    'session': chosen.name.replace('session_', ''),
    'sdir': chosen.name, 'actPort': ACT_PORT,
    'started': data.get('started', ''),
})

HTML = r"""
<div id="app"></div>
<style>
  :root{
    --lime:#c9d400; --lime-dk:#a8b400; --ink:#111417; --ink2:#1b2026;
    --panel:#20262e; --line:#2e353e; --txt:#e8ebee; --mut:#98a1ab;
    --ok:#39c26d; --warn:#e2574c;
  }
  *{box-sizing:border-box; margin:0}
  #app{font-family:-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
       color:var(--txt); background:var(--ink); border-radius:14px; overflow:hidden;
       border:1px solid var(--line)}
  /* ---------- header ---------- */
  .top{display:flex; align-items:center; gap:14px; padding:14px 18px;
       background:var(--ink); border-bottom:1px solid var(--line)}
  .logo{background:var(--lime); color:#111; font-weight:900; font-style:italic;
        letter-spacing:-.5px; padding:5px 12px 5px 10px; font-size:19px;
        clip-path:polygon(0 0,100% 0,92% 100%,0 100%)}
  .prod{font-size:14.5px; font-weight:800; letter-spacing:2.2px; text-transform:uppercase}
  .prod small{display:block; font-size:10.5px; font-weight:600; letter-spacing:1.6px;
              color:var(--mut); margin-top:2px}
  .top .right{margin-left:auto; display:flex; align-items:center; gap:10px}
  .live{display:inline-flex; align-items:center; gap:7px; font-size:11px;
        font-weight:800; letter-spacing:1.5px; color:var(--lime);
        border:1px solid var(--lime); padding:5px 11px; border-radius:3px}
  .live i{width:7px; height:7px; border-radius:50%; background:var(--lime);
          animation:blink 1.4s ease-in-out infinite}
  /* ---------- stat strip ---------- */
  .stats{display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--line)}
  .stat{padding:13px 18px; border-right:1px solid var(--line)}
  .stat:last-child{border-right:none}
  .stat b{font-size:22px; font-weight:800; font-variant-numeric:tabular-nums}
  .stat b.acc{color:var(--lime)}
  .stat span{display:block; font-size:10px; font-weight:700; letter-spacing:1.4px;
             text-transform:uppercase; color:var(--mut); margin-top:3px}
  /* ---------- body ---------- */
  .body{display:flex; gap:0; min-height:430px}
  .mapwrap{flex:1 1 auto; padding:18px; position:relative}
  .toolbar{display:flex; gap:8px; margin-bottom:10px}
  .tbtn{background:var(--ink2); color:var(--mut); border:1px solid var(--line);
        font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
        padding:6px 12px; border-radius:3px; cursor:pointer; transition:all .15s}
  .tbtn:hover{color:var(--txt); border-color:#3d4650}
  .tbtn.on{background:var(--lime); color:#111; border-color:var(--lime)}
  svg{display:block; width:100%; height:auto; user-select:none}
  /* ---------- pins ---------- */
  .pin{cursor:pointer}
  .pin .ring{transform-box:fill-box; transform-origin:center;
             animation:ring 2.6s ease-out infinite}
  .pin .core{transition:transform .15s cubic-bezier(.2,1.4,.4,1)}
  .pin:hover .core{transform:scale(1.18)}
  .pin.sel .core{transform:scale(1.22)}
  .pin.done{opacity:.42}
  .pin.done .ring{animation:none; opacity:0}
  .pin .num{font:800 11px -apple-system,sans-serif; fill:#111}
  .enter{opacity:0; animation:enter .55s cubic-bezier(.2,1.25,.35,1) forwards;
         animation-delay:var(--d)}
  /* ---------- side panel ---------- */
  .side{flex:0 0 265px; border-left:1px solid var(--line); background:var(--ink2);
        padding:16px; display:flex; flex-direction:column; gap:12px}
  .side h3{font-size:11px; font-weight:800; letter-spacing:1.6px;
           text-transform:uppercase; color:var(--mut)}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:8px;
        overflow:hidden; animation:fade .25s ease}
  .card img{width:100%; display:block; border-bottom:1px solid var(--line)}
  .card .cb{padding:12px 13px}
  .card .id{font-size:15px; font-weight:800}
  .card .id em{font-style:normal; color:var(--lime)}
  .kv{display:flex; justify-content:space-between; font-size:12px; margin-top:7px}
  .kv span{color:var(--mut)} .kv b{font-weight:700; font-variant-numeric:tabular-nums}
  .conf{height:4px; background:var(--line); border-radius:2px; margin-top:9px; overflow:hidden}
  .conf i{display:block; height:100%; background:var(--lime); border-radius:2px;
          transition:width .5s ease}
  .hint{color:var(--mut); font-size:12.5px; line-height:1.55; padding:10px 2px}
  .row{display:flex; align-items:center; gap:9px; padding:9px 11px; border-radius:6px;
       cursor:pointer; font-size:12.5px; border:1px solid transparent; transition:all .12s}
  .row:hover{background:var(--panel)}
  .row.sel{background:var(--panel); border-color:var(--line)}
  .row .dotn{width:20px; height:20px; border-radius:50%; background:var(--lime);
             color:#111; font-size:11px; font-weight:800; display:flex;
             align-items:center; justify-content:center; flex:none}
  .row.done .dotn{background:var(--line); color:var(--mut)}
  .row.done{text-decoration:line-through; color:var(--mut)}
  .row b{margin-left:auto; color:var(--mut); font-weight:600; font-size:11.5px}
  .row .rx{flex:none; width:18px; height:18px; border-radius:4px; border:1px solid var(--line);
           color:var(--mut); font-size:11px; line-height:16px; text-align:center;
           opacity:0; transition:all .12s; cursor:pointer}
  .row:hover .rx{opacity:1}
  .row .rx:hover{color:#fff; background:var(--warn); border-color:var(--warn)}
  .actions{display:flex; gap:8px; margin-top:12px}
  .abtn{flex:1; font-size:11px; font-weight:800; letter-spacing:.8px;
        text-transform:uppercase; padding:8px 6px; border-radius:4px;
        border:1px solid var(--line); background:transparent; color:var(--mut);
        cursor:pointer; transition:all .13s}
  .abtn:hover{color:var(--txt); border-color:#3d4650}
  .abtn.primary{background:var(--lime); border-color:var(--lime); color:#111}
  .abtn.primary:hover{background:#d8e21a; color:#111}
  .abtn.danger:hover{background:var(--warn); border-color:var(--warn); color:#fff}
  .foot{padding:10px 18px; border-top:1px solid var(--line); display:flex;
        font-size:10.5px; color:var(--mut); letter-spacing:.4px}
  .foot span+span{margin-left:auto}
  .allclear{display:flex; flex-direction:column; align-items:center; gap:8px;
            justify-content:center; text-align:center; height:100%; color:var(--mut)}
  .allclear .tick{width:44px; height:44px; border-radius:50%; border:2px solid var(--ok);
                  color:var(--ok); font-size:22px; display:flex; align-items:center;
                  justify-content:center; animation:enter .5s ease both}
  .scan{animation:scanmove 5.5s linear infinite}
  .routeline{animation:dashmove 1.2s linear infinite}
  .banner{position:absolute; top:10px; left:50%; transform:translateX(-50%);
          background:var(--lime); color:#111; font-size:12px; font-weight:800;
          letter-spacing:1.6px; text-transform:uppercase; padding:8px 18px;
          border-radius:4px; box-shadow:0 6px 22px rgba(201,212,0,.35);
          animation:bannerin .45s cubic-bezier(.2,1.3,.4,1) both,
                    bannerout .5s ease 3.6s both; z-index:5; white-space:nowrap}
  .banner i{font-style:normal; margin-right:7px}
  @keyframes scanmove{from{transform:translateY(0)} to{transform:translateY(560px)}}
  @keyframes dashmove{from{stroke-dashoffset:18} to{stroke-dashoffset:0}}
  @keyframes bannerin{from{opacity:0; transform:translate(-50%,-16px)}
                      to{opacity:1; transform:translate(-50%,0)}}
  @keyframes bannerout{to{opacity:0; transform:translate(-50%,-10px)}}
  @keyframes ring{0%{transform:scale(.55); opacity:.6} 75%{opacity:0} 100%{transform:scale(2.1); opacity:0}}
  @keyframes blink{0%,100%{opacity:1} 50%{opacity:.25}}
  @keyframes enter{from{opacity:0; transform:translateY(-14px)} to{opacity:1; transform:translateY(0)}}
  @keyframes fade{from{opacity:0} to{opacity:1}}
</style>
<script>
const D = __PAYLOAD__;
const W=D.W, L=D.L, T=D.targets||[];
let sel=null, grid=false, fov=true, route=false;
// survive the live auto-refresh: restore map state if storage is available
const SKEY='sentry_'+D.session;
let prevCount=-1;
try{
  const s=JSON.parse(localStorage.getItem(SKEY)||'{}');
  if(typeof s.sel==='number' && s.sel<T.length) sel=s.sel;
  if(typeof s.grid==='boolean') grid=s.grid;
  if(typeof s.fov==='boolean') fov=s.fov;
  if(typeof s.route==='boolean') route=s.route;
  if(typeof s.count==='number') prevCount=s.count;
}catch(e){}
function saveState(){
  try{localStorage.setItem(SKEY, JSON.stringify({sel,grid,fov,route,count:T.length}));}catch(e){}
}
// ---- helpers ----
function fmtAgo(iso){
  if(!iso) return '';
  const s=((Date.now()-new Date(iso).getTime())/1000)|0;
  if(s<0||isNaN(s)) return '';
  if(s<60) return s+'s ago';
  if(s<3600) return (s/60|0)+'m ago';
  return (s/3600|0)+'h '+((s%3600)/60|0)+'m ago';
}
function fmtTime(iso){
  if(!iso) return '';
  const d=new Date(iso);
  return isNaN(d) ? '' : d.toTimeString().slice(0,8);
}
// nearest-neighbour cleanup route from the mower through OPEN targets
function cleanupRoute(){
  const open=T.map((t,i)=>({...t,i})).filter(t=>!t.cleaned);
  let cur={x_m:W/2, y_m:0}, left=[...open], order=[], dist=0;
  while(left.length){
    let bi=0, bd=1e9;
    left.forEach((t,k)=>{const d=Math.hypot(t.x_m-cur.x_m,t.y_m-cur.y_m);
                          if(d<bd){bd=d; bi=k;}});
    dist+=bd; cur=left[bi]; order.push(cur); left.splice(bi,1);
  }
  return {order, dist};
}
// persist an action via the local endpoint, with an optimistic UI update
function act(action, tid, i){
  try{
    fetch(`http://localhost:${D.actPort}/act?session=${D.sdir}&id=${tid}&action=${action}`)
      .catch(()=>{});
  }catch(e){}
  if(action==='clean')   T[i].cleaned=true;
  if(action==='unclean') T[i].cleaned=false;
  if(action==='dismiss'){T.splice(i,1); if(sel===i) sel=null;
                         else if(sel>i) sel-=1;}
  saveState(); render();
}
// short two-tone chime for a new detection (best effort — may need one click
// in the page first because of browser autoplay rules)
function chime(){
  try{
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    [[880,0],[1318.5,0.14]].forEach(([f,at])=>{
      const o=ac.createOscillator(), g=ac.createGain();
      o.frequency.value=f; o.type='sine'; o.connect(g); g.connect(ac.destination);
      g.gain.setValueAtTime(0.001, ac.currentTime+at);
      g.gain.exponentialRampToValueAtTime(0.12, ac.currentTime+at+0.02);
      g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime+at+0.25);
      o.start(ac.currentTime+at); o.stop(ac.currentTime+at+0.3);
    });
  }catch(e){}
}

const MAPW=640, MAPH=Math.min(520, Math.max(300, MAPW*L/W));
const PAD=26, SX=(MAPW-2*PAD)/W, SY=(MAPH-2*PAD)/L;
const X=x=>PAD+x*SX, Y=y=>MAPH-PAD-y*SY;
const camX=X(W/2), camY=Y(0);

function turf(){
  let s='', n=Math.max(5, Math.round(L*2));
  for(let i=0;i<n;i++){
    const y0=PAD+i*((MAPH-2*PAD)/n), h=(MAPH-2*PAD)/n;
    s+=`<rect x="${PAD}" y="${y0}" width="${MAPW-2*PAD}" height="${h}"
         fill="${i%2?'#3e7a37':'#437f3b'}"/>`;
  }
  return s;
}
function gridLines(){
  if(!grid) return '';
  let s='';
  for(let m=0.5; m<W; m+=0.5)
    s+=`<line x1="${X(m)}" y1="${PAD}" x2="${X(m)}" y2="${MAPH-PAD}"
         stroke="rgba(255,255,255,.13)" stroke-width="1" stroke-dasharray="1 5"/>`;
  for(let m=0.5; m<L; m+=0.5)
    s+=`<line x1="${PAD}" y1="${Y(m)}" x2="${MAPW-PAD}" y2="${Y(m)}"
         stroke="rgba(255,255,255,.13)" stroke-width="1" stroke-dasharray="1 5"/>`;
  return s;
}
function fovCone(){
  if(!fov) return '';
  return `<polygon points="${camX},${camY} ${X(0)},${Y(L)} ${X(W)},${Y(L)}"
          fill="rgba(201,212,0,.05)" stroke="rgba(201,212,0,.22)"
          stroke-width="1" stroke-dasharray="4 5"/>`;
}
function measure(){
  if(sel===null) return '';
  const t=T[sel], mx=X(t.x_m), my=Y(t.y_m);
  const dist=Math.hypot(t.x_m-W/2, t.y_m).toFixed(2);
  const lx=(camX+mx)/2, ly=(camY+my)/2;
  return `<line x1="${camX}" y1="${camY}" x2="${mx}" y2="${my}"
            stroke="#c9d400" stroke-width="1.6" stroke-dasharray="6 5"/>
          <rect x="${lx-26}" y="${ly-11}" width="52" height="18" rx="3" fill="#111417"/>
          <text x="${lx}" y="${ly+3}" text-anchor="middle" fill="#c9d400"
            font-size="11" font-weight="700">${dist} m</text>`;
}
function routePath(){
  if(!route) return '';
  const {order,dist}=cleanupRoute();
  if(!order.length) return '';
  let pts=`${camX},${camY}`;
  order.forEach(t=>{pts+=` ${X(t.x_m)},${Y(t.y_m)}`;});
  const last=order[order.length-1];
  return `<polyline points="${pts}" fill="none" stroke="#c9d400"
            stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"
            class="routeline"/>` +
         order.map((t,k)=>`<circle cx="${X(t.x_m)}" cy="${Y(t.y_m)}" r="17"
            fill="none" stroke="rgba(201,212,0,.35)" stroke-width="1"/>
            <text x="${X(t.x_m)+20}" y="${Y(t.y_m)-14}" fill="#c9d400"
            font-size="10" font-weight="800">${k+1}</text>`).join('') +
         `<text x="${X(last.x_m)}" y="${Y(last.y_m)+30}" text-anchor="middle"
            fill="#c9d400" font-size="10.5" font-weight="700"
            style="letter-spacing:1px">ROUTE ${dist.toFixed(1)} m</text>`;
}
function scanline(){
  if(D.status==='complete') return '';
  return `<g class="scan"><rect x="${PAD}" y="0" width="${MAPW-2*PAD}" height="3"
           fill="rgba(201,212,0,.35)"/></g>`;
}
function pins(){
  return T.map((t,i)=>{
    // clamp into the turf so an edge case can never render off-map
    const cx=Math.min(Math.max(t.x_m,0),W), cy=Math.min(Math.max(t.y_m,0),L);
    const mx=X(cx), my=Y(cy);
    // IMPORTANT: the CSS entry animation lives on an INNER group. Putting an
    // animated CSS transform on the outer <g> would OVERRIDE its SVG
    // translate() attribute and dump every pin at the svg origin (top-left).
    return `<g class="pin ${i===sel?'sel':''} ${t.cleaned?'done':''}"
              data-i="${i}" transform="translate(${mx},${my})">
      <g class="enter" style="--d:${(.12+i*.1).toFixed(2)}s">
        <circle class="ring" r="15" fill="none" stroke="#c9d400" stroke-width="2"/>
        <g class="core">
          <circle r="11.5" fill="#111417" stroke="#c9d400" stroke-width="2.5"/>
          <text class="num" text-anchor="middle" y="4" fill="#c9d400"
                style="fill:#c9d400">${t.target_id}</text>
        </g>
      </g></g>`;
  }).join('');
}
function mapSvg(){
  return `<svg viewBox="0 0 ${MAPW} ${MAPH}">
    <rect x="${PAD-8}" y="${PAD-8}" width="${MAPW-2*PAD+16}" height="${MAPH-2*PAD+16}"
          rx="10" fill="#26302a"/>
    ${turf()}${gridLines()}${fovCone()}
    <g clip-path="url(#turfclip)">${scanline()}</g>
    <clipPath id="turfclip"><rect x="${PAD}" y="${PAD}"
      width="${MAPW-2*PAD}" height="${MAPH-2*PAD}" rx="4"/></clipPath>
    <rect x="${PAD}" y="${PAD}" width="${MAPW-2*PAD}" height="${MAPH-2*PAD}" rx="4"
          fill="none" stroke="#111417" stroke-width="2.5"/>
    ${routePath()}${measure()}
    <g>
      <rect x="${camX-17}" y="${camY-8}" width="34" height="18" rx="3"
            fill="#111417" stroke="#c9d400" stroke-width="1.5"/>
      <rect x="${camX-9}" y="${camY-3}" width="18" height="8" rx="2" fill="#c9d400"/>
      <text x="${camX}" y="${camY+26}" text-anchor="middle" fill="#98a1ab"
            font-size="9.5" font-weight="700" letter-spacing="1.5">MOWER</text>
    </g>
    ${pins()}
    <text x="${PAD}" y="${MAPH-8}" fill="#5c6570" font-size="9.5"
          letter-spacing="1">${W.toFixed(1)} m × ${L.toFixed(1)} m</text>
  </svg>`;
}
function sideHtml(){
  const open=T.filter(t=>!t.cleaned).length;
  if(!T.length)
    return `<div class="allclear"><div class="tick">✓</div>
            <b style="color:var(--txt)">All clear</b>
            <span style="font-size:12px">No waste detected on this lawn</span></div>`;
  let h='';
  if(sel!==null){
    const t=T[sel];
    h+=`<div class="card">
      ${t.photo?`<img src="${t.photo}" alt="detection ${t.target_id}">`:''}
      <div class="cb">
        <div class="id">TARGET <em>#${String(t.target_id).padStart(2,'0')}</em>
          ${t.cleaned?' · <span style="color:var(--ok);font-size:11px">CLEANED</span>':''}</div>
        <div class="kv"><span>Position</span><b>${t.x_m.toFixed(2)} m, ${t.y_m.toFixed(2)} m</b></div>
        <div class="kv"><span>From mower</span><b>${Math.hypot(t.x_m-W/2,t.y_m).toFixed(2)} m</b></div>
        <div class="kv"><span>Detected</span><b>${fmtTime(t.timestamp)}${fmtAgo(t.timestamp)?' · '+fmtAgo(t.timestamp):''}</b></div>
        <div class="kv"><span>Sightings</span><b>${t.observations}</b></div>
        <div class="kv"><span>Confidence</span><b>${(t.confidence*100).toFixed(0)}%</b></div>
        <div class="conf"><i style="width:${(t.confidence*100).toFixed(0)}%"></i></div>
        <div class="actions">
          ${t.cleaned
            ? `<button class="abtn" data-act="unclean" data-tid="${t.target_id}" data-i="${sel}">↺ Re-open</button>`
            : `<button class="abtn primary" data-act="clean" data-tid="${t.target_id}" data-i="${sel}">✓ Mark cleaned</button>`}
          <button class="abtn danger" data-act="dismiss" data-tid="${t.target_id}" data-i="${sel}">✕ Not poop</button>
        </div>
      </div></div>`;
  } else {
    h+=`<div class="hint">Select a target on the map to inspect the camera
        snapshot, exact position and distance from the mower.</div>`;
  }
  h+=`<h3>Detections · ${open} open</h3>`;
  h+=T.map((t,i)=>`<div class="row ${i===sel?'sel':''} ${t.cleaned?'done':''}" data-i="${i}">
        <span class="dotn">${t.target_id}</span>
        <span>${t.x_m.toFixed(2)}, ${t.y_m.toFixed(2)} m
          <span style="color:var(--mut);font-size:10.5px"> · ${fmtAgo(t.timestamp)||fmtTime(t.timestamp)}</span></span>
        <b>${(t.confidence*100).toFixed(0)}%</b>
        <span class="rx" data-act="dismiss" data-tid="${t.target_id}" data-i="${i}"
              title="Not poop — remove">✕</span></div>`).join('');
  return h;
}
function render(){
  const open=T.filter(t=>!t.cleaned).length;
  document.getElementById('app').innerHTML=`
  <div class="top">
    <div class="logo">RYOBI</div>
    <div class="prod">Lawn Sentry<small>Pet waste detection</small></div>
    <div class="right"><span class="live"><i></i>${D.status==='complete'?'SCAN COMPLETE':'LIVE'}</span></div>
  </div>
  <div class="stats">
    <div class="stat"><b class="acc">${open}</b><span>Open targets</span></div>
    <div class="stat"><b>${T.length}</b><span>Total detected</span></div>
    <div class="stat"><b>${(W*L).toFixed(1)}<i style="font-size:13px;font-style:normal"> m²</i></b><span>Area monitored</span></div>
    <div class="stat"><b>${(()=>{const ts=T.map(t=>t.timestamp).filter(Boolean).sort();
        return ts.length?(fmtAgo(ts[ts.length-1])||'now'):'—';})()}</b><span>Last detection</span></div>
  </div>
  <div class="body">
    <div class="mapwrap">
      <div class="toolbar">
        <button class="tbtn ${grid?'on':''}" id="bgrid">Grid 0.5 m</button>
        <button class="tbtn ${fov?'on':''}" id="bfov">Camera FOV</button>
        <button class="tbtn ${route?'on':''}" id="broute">Cleanup route</button>
        <button class="tbtn" id="bclr" ${sel===null?'disabled':''}>Clear selection</button>
      </div>
      ${mapSvg()}
      <div id="alert"></div>
    </div>
    <div class="side">${sideHtml()}</div>
  </div>
  <div class="foot"><span>Session ${D.session}</span><span>SO-101 vision unit · OAK-D</span></div>`;
  // events
  document.querySelectorAll('[data-act]').forEach(el=>{
    el.addEventListener('click',e=>{
      e.stopPropagation();
      act(el.dataset.act, el.dataset.tid, +el.dataset.i);
    });
  });
  document.querySelectorAll('.pin,.row').forEach(el=>{
    el.addEventListener('click',()=>{sel=(+el.dataset.i===sel)?null:+el.dataset.i; saveState(); render();});
  });
  document.getElementById('bgrid').onclick=()=>{grid=!grid; saveState(); render();};
  document.getElementById('bfov').onclick=()=>{fov=!fov; saveState(); render();};
  document.getElementById('broute').onclick=()=>{route=!route; saveState(); render();};
  document.getElementById('bclr').onclick=()=>{sel=null; saveState(); render();};
}
render();
// ---- new-detection alert (fires when the live refresh brings a new target) --
if(prevCount>=0 && T.length>prevCount){
  const n=T.length-prevCount;
  const el=document.getElementById('alert');
  if(el) el.innerHTML=`<div class="banner"><i>●</i>${n>1?n+' new targets':'New target'} detected</div>`;
  chime();
  sel=T.length-1;                       // auto-select the newest find
  saveState(); render();
  const el2=document.getElementById('alert');
  if(el2) el2.innerHTML=`<div class="banner"><i>●</i>${n>1?n+' new targets':'New target'} detected</div>`;
}else{
  saveState();
}
// ---- keyboard: arrows cycle targets, Esc clears -----------------------------
document.addEventListener('keydown',e=>{
  if(!T.length) return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){
    sel=(sel===null)?0:(sel+1)%T.length; saveState(); render();}
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){
    sel=(sel===null)?T.length-1:(sel-1+T.length)%T.length; saveState(); render();}
  else if(e.key==='Escape'){sel=null; saveState(); render();}
});
</script>
"""
components.html(HTML.replace('__PAYLOAD__', PAYLOAD), height=760, scrolling=False)

# ---- mark-as-cleaned (persists to the session folder) ----------------------
if records:
    with st.expander('Mark targets as cleaned'):
        changed = False
        cols = st.columns(min(4, len(records)))
        for i, r in enumerate(records):
            key = str(r['target_id'])
            with cols[i % len(cols)]:
                val = st.checkbox(f"Target #{r['target_id']}",
                                  value=bool(cleaned.get(key)), key=f'cl_{key}')
            if val != bool(cleaned.get(key)):
                cleaned[key] = val
                changed = True
        ca, cb = st.columns(2)
        if ca.button('Mark all cleaned', use_container_width=True):
            for r in records:
                cleaned[str(r['target_id'])] = True
            changed = True
        if cb.button('Reset all', use_container_width=True):
            cleaned = {}
            changed = True
        if changed:
            cleaned_file.write_text(json.dumps(cleaned))
            st.rerun()

if dismissed_records:
    with st.expander(f'Removed as "not poop" ({len(dismissed_records)})'):
        st.caption('These were dismissed from the map. Restore if removed by mistake.')
        for r in dismissed_records:
            key = str(r['target_id'])
            c1, c2 = st.columns([3, 1])
            c1.write(f"Target #{r['target_id']} — {r['x_m']:.2f} m, {r['y_m']:.2f} m "
                     f"({r['confidence']:.0%})")
            if c2.button('Restore', key=f'rs_{key}', use_container_width=True):
                dismissed[key] = False
                dismissed_file.write_text(json.dumps(dismissed))
                st.rerun()
    st.download_button('Download detection report (CSV)',
                       (chosen / 'detections.csv').read_bytes(),
                       'lawn_sentry_report.csv', 'text/csv')

# ---- built-in live refresh (no extra package needed) -----------------------
# While a scan is running, re-read session.json every couple of seconds so new
# detections appear on the map as they happen.
if not _HAVE_AUTOREFRESH and data.get('status') != 'complete':
    import time as _t
    _t.sleep(2.5)
    st.rerun()
