"""Stationary lawn scan, detections to calibrated 2D lawn coordinates.

Uses the existing working OAK-D camera and PoopDetector. Run calibration first.
Keys: [ ] threshold, m write map, s save preview, r start new session,
      q finish and quit.
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
import csv, json, math, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import poop_config as cfg
from trim_detector import build_pipeline
from camera_session import CameraSession
from poop_detector import PoopDetector

from paths import DATA, SESSIONS, ensure as _ensure
_ensure()
HERE = DATA          # data files (calibration, extent) live in data/
CAL_FILE = DATA / 'lawn_calibration.json'
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Stationary OAK-D Lawn Scan'
CONFIRM_HITS = 3
TRACK_TIMEOUT_S = 1.5
# How close (in metres on the lawn) two sightings must be to count as the SAME
# object. Too large and two nearby poops collapse into one pin.
MATCH_M = 0.14   # same live track (was 0.22)
MERGE_M = 0.18   # same already-logged target (was 0.28 — merged poops <28cm apart)
# How far OUTSIDE the calibrated rectangle a detection may land and still be
# accepted (then clamped onto the lawn). Without this, far-edge targets are
# boxed on the video but never reach the map.
EDGE_TOL_M = 0.40

# --- target expiry ---------------------------------------------------------
# A pin disappears from the dashboard if the object is gone. "Gone" has to mean
# LOOKED FOR AND NOT THERE — not merely out of shot. If we expired on time
# alone, panning across the lawn would wipe every target behind the camera.
# So a target is only dropped when it is currently inside the field of view
# (and inside the detection ROI) yet has not been re-detected for TARGET_TTL_S.
TARGET_TTL_S = 3.0
# Grace after first logging it, so a detection that flickers on the very first
# frames is not immediately culled.
TARGET_GRACE_S = 1.5
# Ignore the outermost pixels when deciding "in view" — an object half off the
# edge is not reliably detectable, so it must not count as looked-for.
VIEW_MARGIN_PX = 34
# How often the session file is rewritten for pose/heartbeat updates. Keeps the
# camera icon moving smoothly without writing on every single frame.
WRITE_EVERY_S = 0.35

# --- outline sanity check --------------------------------------------------
# Tracking dead-reckons, so it can slide off the lawn entirely - the outline
# ends up drawn across the floor while the map keeps reporting positions as if
# it were fine. The grass is a strong, cheap signal: if the region inside the
# outline is no longer mostly green, the outline is not on the lawn any more.
# When that happens, snap the tracker back to the calibration, which is the
# same thing the R key does by hand.
OUTLINE_CHECK_EVERY_N = 12     # frames between checks (cheap, but not free)
OUTLINE_MIN_GREEN = 0.35       # fraction of the outline that must be grass
OUTLINE_BAD_STRIKES = 4        # consecutive bad checks before acting
OUTLINE_RECOVER_COOLDOWN_S = 6.0   # do not thrash if the lawn is out of view
# Wipe previous scan sessions when a new run starts, so the dashboard only ever
# shows the current demo. Set False to keep history.
CLEAR_OLD_SESSIONS = True

# --- AUTO LAWN SHAPE (opt-in, OFF by default) ------------------------------
# False -> the demo behaviour you have now: the lawn is the rectangle you
#          clicked in calibrate_lawn_map.py.
# True  -> the camera decides what the lawn is by finding the green area in
#          front of it, and the dashboard draws that real outline instead of a
#          rectangle. Scale still comes from the homography, so you still run
#          calibrate_lawn_map.py once — but the SHAPE is then automatic.
# The LAWN DOES NOT MOVE — the camera does. Re-fitting the lawn outline from
# whatever green is currently visible can never be stable: the mask picks up
# turf beyond the calibrated area and the mower's own green parts, which skews
# the fit and makes the map rotate and rescale wrongly.
#
# So: the lawn is a FIXED rectangle in world coordinates (your calibrated
# dimensions), and the CAMERA is drawn at its live tracked pose. Walk around
# the lawn and the camera icon moves — the lawn and the poops stay put, which
# is what "correct from any viewpoint" actually means.
AUTO_LAWN_SHAPE = False     # no green-fitting; the rectangle is authoritative
LAWN_SHAPE_EVERY_N = 10
SHOW_CAMERA_POSE = True     # publish the live camera position + heading

# --- live video feed for the dashboard -------------------------------------
# The annotated frame is written to the session folder; the web app pushes it
# to the browser over one long-lived MJPEG connection, so the feed runs at
# camera rate independently of the page.
#
# Every frame now, not every other one, and encoded near the size the panel
# actually displays rather than at full sensor resolution. Measured: 0.51 ms
# and 9.4 KB per published frame, against 0.76 ms and 15.3 KB for a full-size
# one — so twice the frame rate costs less per frame in both CPU and bytes.
LIVE_EVERY_N = 1
LIVE_JPEG_Q = 62
LIVE_MAX_W = 512            # downscale before encoding; 0 = full size

# --- decouple the video from the detector ----------------------------------
# The detector is the slow part: with onnxruntime unavailable it falls back to
# cv2.dnn, and 2x2 tiling means five inferences per frame. Running it on every
# frame dragged the WHOLE loop down to detector speed, and the video with it.
# Detections do not need to be recomputed 15 times a second - the targets are
# not going anywhere - so run it every Nth frame and publish video every frame.
DETECT_EVERY_N = 3

# --- MOVING CAMERA (opt-in, OFF by default) --------------------------------
# False -> the demo behaviour: the camera must stay still, because the
#          calibration homography is only true for the pose it was measured at.
# True  -> track the ground between frames so world coordinates stay correct
#          while the camera pans or drives. A poop that hasn't moved keeps the
#          same map position instead of spawning phantom targets.
#          This is the foundation for mounting on the mower.
MOVING_CAMERA = True
# Re-anchoring cancels the drift that dead-reckoning accumulates. Confirmed
# poops are surveyed points, so seeing one again tells us exactly how far the
# estimate has slipped; the lawn border is a coarser fallback when none is in
# view. Corrections are damped, so a bad match nudges rather than teleports.
ANCHOR_TO_LANDMARKS = True
ANCHOR_DAMPING = 0.35        # fraction of the residual applied per correction
ANCHOR_MATCH_M = 0.45

# --- USE THE ARM'S ENCODER INSTEAD OF VISUAL TRACKING ----------------------
# The camera is mounted on the SO-101, which already knows its base angle.
# Visual tracking compares consecutive frames, so its error accumulates: a
# fixed ground point drifts 8 px after 5 degrees of pan and 537 px after 45.
# An encoder reading is absolute, so nothing accumulates and the arm can be
# turned to any angle without the map wandering.
# 'auto' uses the encoder when the servo answers and falls back to visual
# tracking when it does not, so an unplugged arm still runs.
USE_ENCODER = False           # 'auto' | True | False
# Set back to False: this is yesterday's proven visual tracking. The encoder
# version is better in principle and measured well in simulation, but it is
# not yet right on this hardware, and the demo matters more than the theory.

# --- arm control from the dashboard ----------------------------------------
# The scan owns the serial bus, so the web app cannot drive the arm directly.
# It writes a one-line command file instead and this loop executes it - which
# also means a crashed browser can never leave the arm mid-move.
ARM_CONTROL = False           # the DASHBOARD owns the arm now, not the scan.
# Routing button presses through a file that this loop polled cost 1-2 seconds
# per press. With USE_ENCODER off the scan does not need the servos at all, so
# the web app talks to them directly and a press moves the arm immediately.
SURVEY_SPAN_DEG = 50.0        # total width of the automatic survey sweep        # how close a detection must be to count as "the same poop"

def purge_old_sessions():
    """Delete previous scan sessions so the dashboard shows only this run.

    Old sessions pile up and make the session picker look cluttered and
    unprofessional in a demo. Disable with CLEAR_OLD_SESSIONS = False.
    """
    if not CLEAR_OLD_SESSIONS or not SESSIONS.exists():
        return
    import shutil
    n = 0
    for p in SESSIONS.glob('session_*'):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    latest = SESSIONS / 'latest_session.txt'
    if latest.exists():
        try:
            latest.unlink()
        except OSError:
            pass
    if n:
        print(f'[scan] cleared {n} previous session(s) — dashboard starts clean')


class Session:
    def __init__(self, cal):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = SESSIONS / f'session_{stamp}'
        self.snaps = self.dir / 'snapshots'
        self.snaps.mkdir(parents=True, exist_ok=True)
        self.cal = cal; self.start = time.monotonic(); self.targets=[]; self.tracks=[]; self.next_id=1
        self.csv_path = self.dir / 'detections.csv'
        self.json_path = self.dir / 'session.json'
        self.lawn_poly = None          # auto-detected outline, metres (or None)
        self.lawn_area = 0.0
        self.live_H = None             # set each frame when MOVING_CAMERA is on
        self.camera = None             # {'x','y','heading'} live camera pose
        self.seen = {}                 # target_id -> monotonic time last re-detected
        self.arm = None                # live arm status, published to the map
        self.recovered = 0             # times the outline had to be snapped back
        self._last_write = 0.0
        self._write()

    def touch(self, force=False):
        """Rewrite the session file, at most every WRITE_EVERY_S unless forced."""
        now = time.monotonic()
        if force or now - self._last_write >= WRITE_EVERY_S:
            self._last_write = now
            self._write()

    def expire(self, visible, now=None):
        """Drop targets that are in view but have stopped being detected.

        `visible(x_m, y_m)` returns True when that lawn position is currently
        inside the camera's view and detection ROI. Targets outside the view are
        never touched — being out of shot is not evidence the object has gone.
        """
        now = now if now is not None else time.monotonic()
        keep, gone = [], []
        for t in self.targets:
            tid = t['target_id']
            last = self.seen.get(tid)
            if last is None:
                self.seen[tid] = last = now
            if (now - last > max(TARGET_TTL_S, TARGET_GRACE_S)
                    and visible(t['x_m'], t['y_m'])):
                gone.append(t)
            else:
                keep.append(t)
        if gone:
            self.targets = keep
            for t in gone:
                self.seen.pop(t['target_id'], None)
                # forget the live track too, else it re-confirms instantly
                self.tracks = [r for r in self.tracks
                               if math.hypot(r['x_m'] - t['x_m'],
                                             r['y_m'] - t['y_m']) > MERGE_M]
                print(f"[GONE #{t['target_id']}] no longer at "
                      f"x={t['x_m']:.2f}m y={t['y_m']:.2f}m — removed from map")
            self.touch(force=True)
        return gone

    def set_camera(self, pose):
        """Record where the camera currently is, in lawn metres."""
        if pose is None:
            return
        x, y, hd = pose
        self.camera = {'x_m': round(x, 3), 'y_m': round(y, 3),
                       'heading_deg': round(hd, 1)}

    def set_lawn_polygon(self, poly, area):
        """Store an auto-detected lawn outline; the dashboard draws this shape."""
        self.lawn_poly = poly
        self.lawn_area = area
        self._write()

    def project(self, u, v):
        """Image pixel -> lawn metres.

        Uses the live tracked homography when the camera is moving, otherwise
        the fixed calibration."""
        H = self.live_H if self.live_H is not None else \
            np.asarray(self.cal['homography_image_to_lawn'], np.float64)
        p=cv2.perspectiveTransform(np.array([[[float(u),float(v)]]],np.float32),H)[0,0]
        return float(p[0]),float(p[1])

    @staticmethod
    def bottom_centre(d):
        x,y,w,h=d['bbox']; return x+w/2.0, y+h

    def _nearest(self, x, y, entries, radius):
        good=[e for e in entries if math.hypot(e['x_m']-x,e['y_m']-y)<=radius]
        return min(good,key=lambda e: math.hypot(e['x_m']-x,e['y_m']-y)) if good else None

    def update(self, dets, clean_frame):
        now=time.monotonic()
        self.tracks=[t for t in self.tracks if now-t['last']<=TRACK_TIMEOUT_S]
        newly=[]
        LW=self.cal['lawn_width_m']; LL=self.cal['lawn_length_m']
        for d in dets:
            u,v=self.bottom_centre(d); x,y=self.project(u,v)
            # The lawn rectangle is a CALIBRATION, not a physical wall. A poop
            # sitting on the far edge projects to y ~= LL, and a few cm of
            # homography error then pushes it outside and it silently vanishes
            # from the map (while still being boxed on the video). Allow a
            # margin and clamp onto the lawn instead of discarding.
            if self.lawn_poly:
                # Auto-detected outline. Be GENEROUS: the detector already
                # rejected anything off the grass in image space, and a brief
                # lapse in green segmentation must never start throwing away
                # real detections. Inside the outline, or within EDGE_TOL_M of
                # its bounding box, both count.
                import lawn_shape as _ls
                if not _ls.point_in_polygon(self.lawn_poly, x, y):
                    mnx,mny,mxx,mxy=_ls.polygon_bounds(self.lawn_poly)
                    if not (mnx-EDGE_TOL_M<=x<=mxx+EDGE_TOL_M
                            and mny-EDGE_TOL_M<=y<=mxy+EDGE_TOL_M):
                        continue
            else:
                if not (-EDGE_TOL_M<=x<=LW+EDGE_TOL_M and -EDGE_TOL_M<=y<=LL+EDGE_TOL_M):
                    continue
                x=min(max(x,0.0),LW); y=min(max(y,0.0),LL)
            # Any detection landing on an already-logged target counts as
            # "still there" — this is what keeps its pin alive.
            near=self._nearest(x,y,self.targets,MERGE_M)
            if near is not None:
                self.seen[near['target_id']]=now
            tr=self._nearest(x,y,self.tracks,MATCH_M)
            if tr is None:
                tr={'x_m':x,'y_m':y,'hits':1,'last':now,'best':dict(d),'score':float(d['score'])}
                self.tracks.append(tr)
            else:
                a=.35; tr['x_m']=(1-a)*tr['x_m']+a*x; tr['y_m']=(1-a)*tr['y_m']+a*y
                tr['hits']+=1; tr['last']=now
                if float(d['score'])>tr['score']: tr['best']=dict(d); tr['score']=float(d['score'])
            if tr['hits']>=CONFIRM_HITS and not tr.get('done'):
                old=self._nearest(tr['x_m'],tr['y_m'],self.targets,MERGE_M)
                if old is None:
                    tid=self.next_id; self.next_id+=1
                    snap=f'target_{tid:03d}.jpg'; annotated=clean_frame.copy()
                    bx,by,bw,bh=tr['best']['bbox']
                    cv2.rectangle(annotated,(bx,by),(bx+bw,by+bh),(0,0,255),3)
                    cv2.putText(annotated,f'Poop #{tid}',(bx,max(18,by-8)),FONT,.6,(0,0,255),2)
                    # q85 instead of the default 95: visually identical for a
                    # thumbnail, roughly a third of the bytes to store, serve
                    # and (as a fallback) embed in the page.
                    cv2.imwrite(str(self.snaps/snap),annotated,
                                [int(cv2.IMWRITE_JPEG_QUALITY),85])
                    obj={'target_id':tid,'timestamp':datetime.now().isoformat(timespec='milliseconds'),
                         'x_m':round(tr['x_m'],3),'y_m':round(tr['y_m'],3),'confidence':round(tr['score'],4),
                         'observations':tr['hits'],'snapshot':f'snapshots/{snap}','status':'detected'}
                    self.targets.append(obj); newly.append(obj); self.seen[tid]=now
                    print(f"[POOP #{tid}] x={obj['x_m']:.2f}m y={obj['y_m']:.2f}m confidence={obj['confidence']:.2f}")
                else:
                    self.seen[old['target_id']]=now
                tr['done']=True
        if newly: self._write()
        return newly

    def _write(self):
        self.dir.mkdir(parents=True,exist_ok=True)
        fields=['target_id','timestamp','x_m','y_m','confidence','observations','snapshot','status']
        with self.csv_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(self.targets)
        payload={'session_id':self.dir.name,'status':'live','started':datetime.now().isoformat(timespec='seconds'),
                 'lawn_width_m':self.cal['lawn_width_m'],'lawn_length_m':self.cal['lawn_length_m'],
                 'target_count':len(self.targets),'targets':self.targets}
        if self.lawn_poly:
            # real measured outline — the dashboard draws THIS instead of a rect
            payload['lawn_polygon']=[[round(p[0],3),round(p[1],3)] for p in self.lawn_poly]
            payload['lawn_area_m2']=round(self.lawn_area,2)
        if self.camera:
            payload['camera']=self.camera        # live pose, drawn on the map
        if getattr(self,'arm',None):
            payload['arm']=self.arm              # lets the dashboard show controls
        if getattr(self,'recovered',0):
            payload['recovered']=self.recovered  # outline had to be re-anchored
        temp=self.json_path.with_suffix('.tmp'); temp.write_text(json.dumps(payload,indent=2)); temp.replace(self.json_path)
        (SESSIONS/'latest_session.txt').write_text(self.dir.name)

    def finish(self):
        self._write(); data=json.loads(self.json_path.read_text()); data['status']='complete'; data['finished']=datetime.now().isoformat(timespec='seconds')
        self.json_path.write_text(json.dumps(data,indent=2))

EXTENT_FILE = HERE / 'lawn_extent.json'


def apply_lawn_extent(cal):
    """If define_lawn.py has surveyed the real lawn, adopt its frame.

    calibrate_lawn_map.py fixes the SCALE from one view; define_lawn.py walks
    the camera round and captures the true corners, then gives us a transform
    that makes the lawn axis-aligned with 0,0 at the near-left. Composing that
    with the calibration homography means every detection lands in real lawn
    coordinates, and the outline sits on the actual edges.
    """
    if not EXTENT_FILE.exists():
        return cal, False
    try:
        ext = json.loads(EXTENT_FILE.read_text())
        T = np.asarray(ext['to_lawn_frame'], np.float64)
        H = np.asarray(cal['homography_image_to_lawn'], np.float64)
        Hn = T @ H
        Hn /= Hn[2, 2]
        cal = dict(cal)
        cal['homography_image_to_lawn'] = Hn.tolist()
        cal['lawn_width_m'] = ext['lawn_width_m']
        cal['lawn_length_m'] = ext['lawn_length_m']
        print(f"[scan] using surveyed lawn extent: "
              f"{ext['lawn_width_m']:.2f} x {ext['lawn_length_m']:.2f} m "
              f"(corner fit {ext.get('fit_residual_m',0)*100:.0f} cm)")
        return cal, True
    except (OSError, ValueError, KeyError) as e:
        print(f'[scan] lawn_extent.json unusable ({e}); using calibration only')
        return cal, False


def main():
    if not CAL_FILE.exists(): raise SystemExit('Run calibrate_lawn_map.py first.')
    # via cal_store, which fills in fields a newer calibrator does not produce
    # (the QR method solves the plane outright and never clicks corners)
    import cal_store
    cal=cal_store.load(CAL_FILE); purge_old_sessions()
    cal, _surveyed = apply_lawn_extent(cal)
    cam=CameraSession(build_pipeline); det=PoopDetector(); session=Session(cal); preview_n=0
    print('session:',session.dir); print('web app: streamlit run lawn_map_app.py')
    shape=None; H=None; frame_n=0; tracker=None
    _live_ok=[False,False]        # [logged success, logged failure] — once each
    _bad_outline=0                # consecutive frames the outline was off-grass
    _last_recover=0.0
    base_quad=np.asarray(cal['image_points_tl_tr_br_bl'],np.float32)
    anchor=None
    # Open the arm ONCE here; both the encoder tracker and the control pad
    # borrow this same connection.
    _shared_arm=None
    # Open the arm if ANY of these need it - including simply returning to the
    # calibrated rest pose. This was gated on the encoder/control flags, so
    # with both off the arm was never opened and the return silently did
    # nothing: the map was drawn from wherever the arm happened to be.
    # The arm is handled by launch.py BEFORE any of this starts: it moves to
    # the calibrated rest pose, then releases the serial port so the dashboard
    # can own it. Doing it here would have meant waiting six seconds without
    # draining the camera queue (which faults the OAK-D) and fighting the web
    # app for the same port.
    _needs_arm = False
    if MOVING_CAMERA:
        from ground_track import GroundTracker, LandmarkAnchor
        H0=np.asarray(cal['homography_image_to_lawn'],np.float64)
        tracker=GroundTracker(H0, roi_poly=base_quad)
        if USE_ENCODER:
            try:
                from pan_source import open_pan
                from camera_intrinsics import get as get_K
                from encoder_track import EncoderTracker, ground_normal_from_homography
                if _shared_arm is None:
                    raise RuntimeError('arm not connected')
                _pan=open_pan('arm' if USE_ENCODER is True else 'auto',
                              arm=_shared_arm)
                if _pan.name=='fixed':
                    raise RuntimeError('base servo not readable')
                _K,_ks=get_K(cfg.CAMERA_WIDTH if hasattr(cfg,'CAMERA_WIDTH') else 640,
                             cfg.CAMERA_HEIGHT if hasattr(cfg,'CAMERA_HEIGHT') else 480,
                             quiet=True)
                _n=ground_normal_from_homography(H0,_K)
                enc=EncoderTracker(H0,_K,_pan,roi_poly=base_quad,visual=tracker)
                if not enc.axis_ok:
                    raise RuntimeError('could not read the ground plane from the '
                                       'calibration')
                tracker=enc
                print(f'[scan] ENCODER TRACKING on — base angle from the SO-101 '
                      f'({_ks}).')
                print(f'       Turn the arm to any angle; the map will not drift.')
                print(f'       base now at {_pan.angle_deg():+.1f} deg, '
                      f'ground normal {np.round(_n,3)}')
            except Exception as _e:
                if USE_ENCODER is True:
                    print(f'[scan] encoder tracking unavailable: {_e}')
                print('[scan] falling back to VISUAL ground tracking (drifts on '
                      'long sweeps — keep rotations modest)')
        if not isinstance(tracker, GroundTracker) or USE_ENCODER is False:
            pass
        print('[scan] MOVING CAMERA on — world positions stay correct while the '
              'camera moves')
        if ANCHOR_TO_LANDMARKS:
            anchor=LandmarkAnchor(match_radius_m=ANCHOR_MATCH_M,
                                  damping=ANCHOR_DAMPING)
            print('[scan] landmark re-anchoring on — confirmed poops and the '
                  'lawn border cancel accumulated drift')
    # --- arm control (optional) -------------------------------------------
    # ONE serial connection, shared. Opening the bus twice makes the servos
    # return empty reads ("multiple access on port"), which looks like flaky
    # hardware but is entirely self-inflicted.
    armc=None
    if ARM_CONTROL:
        try:
            from arm_control import ArmControl
            armc=ArmControl(arm=_shared_arm)
            if not armc.ok:
                raise RuntimeError(armc.error or 'arm not responding')
            print('[scan] ARM CONTROL on — use the web app to pan/tilt, or press')
            print('       W to start the slow survey sweep, X to stop it.')
        except Exception as _e:
            armc=None
            if ARM_CONTROL is True:
                print(f'[scan] arm control unavailable: {_e}')
    session.armc=armc

    lawnmap=None; rectlawn=None
    if AUTO_LAWN_SHAPE:
        import lawn_shape as _ls
        shape=_ls.LawnShape()
        lawnmap=_ls.LawnMap()          # accumulates the lawn as the camera pans
        H=np.asarray(cal['homography_image_to_lawn'],np.float64)
        if LAWN_RECTANGLE:
            size=(cal['lawn_width_m'], cal['lawn_length_m']) if LAWN_FIXED_SIZE else None
            rectlawn=_ls.RectLawn(size=size, alpha=LAWN_ANGLE_SMOOTH)
            print(f'[scan] AUTO LAWN on — rectangle '
                  f'{"(fixed %.1fx%.1f m)"%size if size else "(size measured)"}, '
                  'orientation follows the camera')
        else:
            print('[scan] AUTO LAWN SHAPE on — free-form green outline')
    try:
        while True:
            g=cam.get()
            if g is None:
                if cam.crashed() and not cam.restart(): break
                if cv2.waitKey(1)&255==ord('q'): break
                continue
            frame=g['rgb'].getCvFrame(); depth=g['depth'].getFrame().astype(np.float32)/1000.0; stats={}
            frame_n+=1
            roi_quad=cal['image_points_tl_tr_br_bl']
            # --- update the pose FIRST, so the ROI below uses this frame ---
            if tracker is not None:
                session.live_H = tracker.update(frame)
                if SHOW_CAMERA_POSE and frame_n % 5 == 1:
                    from ground_track import camera_pose
                    session.set_camera(camera_pose(session.live_H,
                                                   frame.shape[1], frame.shape[0]))
            h_,w_=frame.shape[:2]
            if _surveyed and session.live_H is not None:
                # We know the REAL lawn rectangle in world metres, so project it
                # into the current view. This is the actual lawn boundary — not
                # the slice the camera happened to see during calibration — so
                # the outline sits on the true edges from any viewpoint.
                try:
                    LWm=cal['lawn_width_m']; LLm=cal['lawn_length_m']
                    world=np.array([[[0,0]],[[LWm,0]],[[LWm,LLm]],[[0,LLm]]],np.float32)
                    proj=cv2.perspectiveTransform(
                        world, np.linalg.inv(session.live_H)).reshape(-1,2)
                    if np.isfinite(proj).all() and np.abs(proj).max()<1e5:
                        inside=sum(1 for p in proj
                                   if -w_<p[0]<2*w_ and -h_<p[1]<2*h_)
                        roi_quad = proj.tolist() if inside>=2 else None
                except (np.linalg.LinAlgError, ValueError):
                    pass
            elif tracker is not None:
                # No survey: warp the originally clicked quad so it follows the
                # ground as the camera moves.
                try:
                    inv=np.linalg.inv(tracker.G)
                    warped=cv2.perspectiveTransform(
                        base_quad.reshape(1,-1,2), inv).reshape(-1,2)
                    # If the camera has turned so far that the calibrated quad
                    # has left the frame, a warped ROI would reject EVERYTHING.
                    # Fall back to the whole frame rather than going blind.
                    inside=sum(1 for p in warped if -w_ < p[0] < 2*w_ and -h_ < p[1] < 2*h_)
                    roi_quad = warped.tolist() if inside>=3 else None
                except np.linalg.LinAlgError:
                    pass
            grass_mask=None
            if shape is not None and frame_n % LAWN_SHAPE_EVERY_N == 1:
                import lawn_shape as _ls
                grass_mask=_ls.green_mask(frame)
                # project through the LIVE pose so the accumulated map stays
                # correct while the camera turns
                Hnow = session.live_H if session.live_H is not None else H
                lawnmap.add(grass_mask, Hnow)
                lp = lawnmap.polygon()
                if lp:
                    if rectlawn is not None:
                        # fit the rectangle to everything seen so far; only its
                        # centre and angle move, so the map never wobbles
                        rl = rectlawn.update(lp)
                        if rl:
                            session.set_lawn_polygon(rl, rectlawn.area())
                            if anchor is not None:
                                anchor.set_lawn_reference(rl)
                    else:
                        session.set_lawn_polygon(lp, lawnmap.area())
                        if anchor is not None:
                            anchor.set_lawn_reference(lp)
            # Publish a frame FIRST, every time round the loop, so the stream
            # runs at camera rate instead of detector rate. The annotated
            # version replaces it a moment later on detection frames.
            if frame_n % LIVE_EVERY_N == 0:
                try:
                    _q=frame
                    if LIVE_MAX_W and frame.shape[1]>LIVE_MAX_W:
                        _sc=LIVE_MAX_W/frame.shape[1]
                        _q=cv2.resize(frame,None,fx=_sc,fy=_sc,
                                      interpolation=cv2.INTER_LINEAR)
                    _t=session.dir/'live.tmp.jpg'
                    cv2.imwrite(str(_t),_q,[int(cv2.IMWRITE_JPEG_QUALITY),LIVE_JPEG_Q])
                    _t.replace(session.dir/'live.jpg')
                except Exception:
                    pass

            # Only run the detector every Nth frame. `dets` persists between
            # runs so the boxes stay on screen and tracking is unaffected.
            if frame_n % DETECT_EVERY_N == 0 or not hasattr(main,'_dets'):
                dets=det.detect(frame,depth,stats,roi_poly=roi_quad)
                main._dets=dets
            else:
                dets=main._dets
            # Re-anchor BEFORE logging: correct the pose using landmarks we
            # already trust, so this frame's detections land in the right place.
            if anchor is not None and session.targets:
                anchor.update(tracker, dets, session.targets, frame_n,
                              lawn_poly=session.lawn_poly)
                session.live_H = tracker.W
            session.update(dets,frame)

            # --- retire targets that are looked at but no longer there -------
            # Project the target's lawn position back into THIS view. If it
            # lands well inside the frame and inside the detection ROI, the
            # detector had a fair chance to see it; if it also hasn't been seen
            # for TARGET_TTL_S, the object has been picked up or moved.
            if session.targets:
                Hvis = session.live_H if session.live_H is not None else \
                    np.asarray(cal['homography_image_to_lawn'],np.float64)
                try:
                    Hinv=np.linalg.inv(Hvis)
                except np.linalg.LinAlgError:
                    Hinv=None
                roi_np=(np.asarray(roi_quad,np.float32)
                        if roi_quad is not None else None)
                def _visible(mx,my,_Hinv=Hinv,_roi=roi_np,_w=w_,_h=h_):
                    if _Hinv is None:
                        return False
                    try:
                        p=cv2.perspectiveTransform(
                            np.array([[[float(mx),float(my)]]],np.float32),_Hinv)[0,0]
                    except cv2.error:
                        return False
                    px,py=float(p[0]),float(p[1])
                    if not (math.isfinite(px) and math.isfinite(py)):
                        return False
                    m=VIEW_MARGIN_PX
                    if not (m<=px<=_w-m and m<=py<=_h-m):
                        return False        # off screen / at the very edge
                    if _roi is not None and \
                            cv2.pointPolygonTest(_roi,(px,py),False)<0:
                        return False        # outside the area we actually search
                    return True
                session.expire(_visible)
            # --- arm commands from the dashboard ---------------------------
            if armc is not None and frame_n % 10 == 0:
                cmdf=session.dir/'arm_cmd.json'
                try:
                    if cmdf.is_file():
                        c=json.loads(cmdf.read_text()); cmdf.unlink()
                        a=c.get('cmd')
                        if a=='look':
                            # Lawn point -> where it falls in the picture ->
                            # how far the camera must turn to centre it. Uses
                            # the LIVE homography, so it stays right after
                            # panning away from the rest pose.
                            try:
                                Hn = session.live_H if session.live_H is not None \
                                    else np.asarray(cal['homography_image_to_lawn'],np.float64)
                                pix = cv2.perspectiveTransform(
                                    np.array([[[float(c.get('x',0)),float(c.get('y',0))]]],np.float32),
                                    np.linalg.inv(Hn))[0,0]
                                if np.isfinite(pix).all():
                                    from camera_intrinsics import get as _gk
                                    _K,_ = _gk(frame.shape[1], frame.shape[0], quiet=True)
                                    dx = float(pix[0]) - _K[0,2]
                                    ang = math.degrees(math.atan2(dx, _K[0,0]))
                                    ang = max(-45.0, min(45.0, ang))
                                    if abs(ang) > 0.5:
                                        armc.act('right', abs(ang)) if ang > 0 \
                                            else armc.act('left', abs(ang))
                                        print(f'  looking at ({c.get("x"):.2f},'
                                              f'{c.get("y"):.2f}) m -> pan {ang:+.1f} deg')
                            except Exception as _e:
                                print(f'  look failed: {_e}')
                        elif a in armc.ACTIONS: armc.act(a, c.get('deg',4.0))
                        elif a=='sweep':   armc.start_sweep(SURVEY_SPAN_DEG)
                        elif a=='stop':    armc.stop_sweep()
                        elif a=='home':    armc.go_home()
                        elif a=='ping':    armc.ping()
                except Exception:
                    pass
                session.arm=armc.status()

            # --- a HOME press from the dashboard resets the plane -----------
            # Deliberately delayed: the file carries the time to act, set a few
            # seconds ahead so the arm has finished moving. Re-anchoring early
            # would capture the view mid-swing and be worse than not doing it.
            if tracker is not None and frame_n % 4 == 0:
                _rq=session.dir/'reanchor.json'
                try:
                    if _rq.is_file():
                        _c=json.loads(_rq.read_text())
                        if time.time() >= float(_c.get('at', 0)):
                            _rq.unlink()
                            tracker.anchor()
                            session.live_H=tracker.W
                            _bad_outline=0
                            _last_recover=time.monotonic()
                            print('[scan] HOME pressed - arm back at rest, lawn '
                                  'plane re-aligned to the calibration.')
                except Exception:
                    pass

            # --- is the outline still ON the lawn? -------------------------
            if (tracker is not None and roi_quad is not None
                    and frame_n % OUTLINE_CHECK_EVERY_N == 0):
                try:
                    import lawn_shape as _ls
                    poly=np.asarray(roi_quad,np.int32)
                    m=np.zeros(frame.shape[:2],np.uint8)
                    cv2.fillConvexPoly(m,poly,255)
                    inside=int((m>0).sum())
                    if inside>2000:
                        green=_ls.green_mask(frame)
                        frac=float((green & (m>0)).sum())/inside
                        if frac < OUTLINE_MIN_GREEN:
                            _bad_outline+=1
                        else:
                            _bad_outline=0
                        if (_bad_outline>=OUTLINE_BAD_STRIKES and
                                time.monotonic()-_last_recover>OUTLINE_RECOVER_COOLDOWN_S):
                            print(f'[scan] outline is only {frac*100:.0f}% on grass '
                                  f'- it has drifted off the lawn. Snapping back '
                                  f'to the calibration.')
                            tracker.anchor()
                            session.live_H=tracker.W
                            _bad_outline=0
                            _last_recover=time.monotonic()
                            session.recovered=session.recovered+1
                except Exception:
                    pass

            session.touch()          # keeps the camera pose live on the map

            display=frame.copy()
            if grass_mask is not None:
                # tint the detected lawn so you can see what it decided
                display[grass_mask]=(0.75*display[grass_mask]
                                     +0.25*np.array([60,190,60])).astype(np.uint8)
            for d in dets:
                x,y,w,h=d['bbox']; cv2.rectangle(display,(x,y),(x+w,y+h),(0,170,255),2)
                px,py=session.project(x+w/2,y+h); cv2.putText(display,f'{px:.2f},{py:.2f}m',(x,max(15,y-6)),FONT,.45,(0,255,255),1)
            if roi_quad is not None:
                poly=np.asarray(roi_quad,np.int32); cv2.polylines(display,[poly],True,(80,255,80),2)
            cv2.rectangle(display,(0,0),(display.shape[1],48),(15,15,15),-1)
            if tracker is not None:
                s=tracker.status()
                if s.get('source'):
                    txt=(f"TRACKING {s['source'].upper()}"
                         f"{' LOST' if s['lost'] else ''}"
                         f"  base={s.get('angle_deg') if s.get('angle_deg') is None else round(s['angle_deg'],1)} deg"
                         f"  moved={s.get('moved_deg',0):.0f} deg")
                else:
                    txt=f"TRACKING {'LOST' if s['lost'] else 'OK'}  feats={s['tracked']}  moved={s['moved_m']:.2f}m"
                if anchor is not None:
                    a=anchor.status()
                    txt+=f"  | anchors={a['corrections']} resid={a['last_residual_cm']:.0f}cm"
                cv2.putText(display,txt,(8,66),FONT,.45,
                            (0,220,90) if not s['lost'] else (0,120,255),1)
            cv2.putText(display,f"targets={len(session.targets)} candidates={len(dets)} threshold={cfg.POOP_SCORE_MIN:.2f}",(8,20),FONT,.5,(0,255,255),1)
            cv2.putText(display,'[ ] threshold | R re-snap | W sweep | X stop | S preview | Q finish',
                        (8,41),FONT,.47,(80,255,80),1)
            # publish the annotated frame for the dashboard's live feed.
            # written atomically so the web app never reads a half-file.
            if frame_n % LIVE_EVERY_N == 0:
                try:
                    out=display
                    if LIVE_MAX_W and display.shape[1]>LIVE_MAX_W:
                        s=LIVE_MAX_W/display.shape[1]
                        # INTER_LINEAR, not INTER_AREA: half the cost and the
                        # difference is invisible at this scale factor
                        out=cv2.resize(display,None,fx=s,fy=s,
                                       interpolation=cv2.INTER_LINEAR)
                    tmp=session.dir/'live.tmp.jpg'
                    cv2.imwrite(str(tmp),out,[int(cv2.IMWRITE_JPEG_QUALITY),LIVE_JPEG_Q])
                    tmp.replace(session.dir/'live.jpg')
                    if not _live_ok[0]:
                        _live_ok[0]=True
                        print(f'  live feed publishing -> {session.dir.name}/live.jpg')
                except Exception as e:
                    if not _live_ok[1]:
                        _live_ok[1]=True
                        print(f'  live feed FAILED to write: {e}')
            cv2.imshow(WIN,display); k=cv2.waitKey(1)&255
            if k==ord('q'): break
            if k==ord('['): cfg.POOP_SCORE_MIN=max(.1,cfg.POOP_SCORE_MIN-.05)
            if k==ord(']'): cfg.POOP_SCORE_MIN=min(.95,cfg.POOP_SCORE_MIN+.05)
            if k==ord('w') and armc is not None:
                print('  survey sweep', 'started' if armc.start_sweep(SURVEY_SPAN_DEG) else 'could not start')
            if k==ord('x') and armc is not None:
                armc.stop_sweep(); print('  survey sweep stopped')
            if k==ord('r') and tracker is not None:
                # Snap the outline back to the calibration. Tracking dead-reckons,
                # so if the green quad has crept off the real lawn edges this
                # resets the accumulated error to zero for the CURRENT view.
                tracker.anchor(); session.live_H=tracker.W
                print('[scan] tracking re-anchored — outline snapped back to calibration')
            if k==ord('s'):
                p=session.dir/f'preview_{preview_n:03d}.jpg'; preview_n+=1; cv2.imwrite(str(p),display); print('saved',p)
    finally:
        # stop the arm FIRST - never leave it sweeping after the window closes
        if 'armc' in dir() and armc is not None:
            try:
                armc.stop_sweep()
                print('[scan] arm stopped')
            except Exception:
                pass
        if '_shared_arm' in dir() and _shared_arm is not None:
            try:
                _shared_arm.close()
                print('[scan] arm connection closed')
            except Exception:
                pass
        session.finish(); cam.stop(); cv2.destroyAllWindows(); print('complete:',session.dir)
if __name__=='__main__': main()
