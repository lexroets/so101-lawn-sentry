"""Stationary lawn scan, detections to calibrated 2D lawn coordinates.

Uses the existing working OAK-D camera and PoopDetector. Run calibration first.
Keys: [ ] threshold, m write map, s save preview, r start new session,
      q finish and quit.
"""
import csv, json, math, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import poop_config as cfg
from trim_detector import build_pipeline
from camera_session import CameraSession
from poop_detector import PoopDetector

HERE = Path(__file__).resolve().parent
CAL_FILE = HERE / 'lawn_calibration.json'
SESSIONS = HERE / 'scan_sessions'
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Stationary OAK-D Lawn Scan'
CONFIRM_HITS = 3
TRACK_TIMEOUT_S = 1.5
MATCH_M = 0.22
MERGE_M = 0.28

class Session:
    def __init__(self, cal):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = SESSIONS / f'session_{stamp}'
        self.snaps = self.dir / 'snapshots'
        self.snaps.mkdir(parents=True)
        self.cal = cal; self.start = time.monotonic(); self.targets=[]; self.tracks=[]; self.next_id=1
        self.csv_path = self.dir / 'detections.csv'
        self.json_path = self.dir / 'session.json'
        self._write()

    def project(self, u, v):
        H=np.asarray(self.cal['homography_image_to_lawn'], np.float64)
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
        for d in dets:
            u,v=self.bottom_centre(d); x,y=self.project(u,v)
            if not (0<=x<=self.cal['lawn_width_m'] and 0<=y<=self.cal['lawn_length_m']): continue
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
                    cv2.imwrite(str(self.snaps/snap),annotated)
                    obj={'target_id':tid,'timestamp':datetime.now().isoformat(timespec='milliseconds'),
                         'x_m':round(tr['x_m'],3),'y_m':round(tr['y_m'],3),'confidence':round(tr['score'],4),
                         'observations':tr['hits'],'snapshot':f'snapshots/{snap}','status':'detected'}
                    self.targets.append(obj); newly.append(obj); print(f"[POOP #{tid}] x={obj['x_m']:.2f}m y={obj['y_m']:.2f}m confidence={obj['confidence']:.2f}")
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
        temp=self.json_path.with_suffix('.tmp'); temp.write_text(json.dumps(payload,indent=2)); temp.replace(self.json_path)
        (SESSIONS/'latest_session.txt').write_text(self.dir.name)

    def finish(self):
        self._write(); data=json.loads(self.json_path.read_text()); data['status']='complete'; data['finished']=datetime.now().isoformat(timespec='seconds')
        self.json_path.write_text(json.dumps(data,indent=2))

def main():
    if not CAL_FILE.exists(): raise SystemExit('Run calibrate_lawn_map.py first.')
    cal=json.loads(CAL_FILE.read_text()); cam=CameraSession(build_pipeline); det=PoopDetector(); session=Session(cal); preview_n=0
    print('session:',session.dir); print('web app: streamlit run lawn_map_app.py')
    try:
        while True:
            g=cam.get()
            if g is None:
                if cam.crashed() and not cam.restart(): break
                if cv2.waitKey(1)&255==ord('q'): break
                continue
            frame=g['rgb'].getCvFrame(); depth=g['depth'].getFrame().astype(np.float32)/1000.0; stats={}
            dets=det.detect(frame,depth,stats,roi_poly=cal['image_points_tl_tr_br_bl']); session.update(dets,frame); display=frame.copy()
            for d in dets:
                x,y,w,h=d['bbox']; cv2.rectangle(display,(x,y),(x+w,y+h),(0,170,255),2)
                px,py=session.project(x+w/2,y+h); cv2.putText(display,f'{px:.2f},{py:.2f}m',(x,max(15,y-6)),FONT,.45,(0,255,255),1)
            poly=np.asarray(cal['image_points_tl_tr_br_bl'],np.int32); cv2.polylines(display,[poly],True,(80,255,80),2)
            cv2.rectangle(display,(0,0),(display.shape[1],48),(15,15,15),-1)
            cv2.putText(display,f"targets={len(session.targets)} candidates={len(dets)} threshold={cfg.POOP_SCORE_MIN:.2f}",(8,20),FONT,.5,(0,255,255),1)
            cv2.putText(display,'[ ] threshold | S preview | Q finish',(8,41),FONT,.47,(80,255,80),1)
            cv2.imshow(WIN,display); k=cv2.waitKey(1)&255
            if k==ord('q'): break
            if k==ord('['): cfg.POOP_SCORE_MIN=max(.1,cfg.POOP_SCORE_MIN-.05)
            if k==ord(']'): cfg.POOP_SCORE_MIN=min(.95,cfg.POOP_SCORE_MIN+.05)
            if k==ord('s'):
                p=session.dir/f'preview_{preview_n:03d}.jpg'; preview_n+=1; cv2.imwrite(str(p),display); print('saved',p)
    finally:
        session.finish(); cam.stop(); cv2.destroyAllWindows(); print('complete:',session.dir)
if __name__=='__main__': main()
