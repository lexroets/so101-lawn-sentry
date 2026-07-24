"""Operations layer — session logging, live metrics, failure detection.

Turns the raw log stream into something you can run unattended and review
afterwards: every run is captured to a timestamped file, key events are
counted, a JSON summary is written at exit, and missed grabs are flagged the
instant they happen instead of silently dropping nothing into a bucket.

Nothing here touches the robot logic — it observes the log and wraps the
gripper read. main.py is untouched.
"""
import json
import os
import re
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, 'session_logs')


class SessionMetrics:
    def __init__(self):
        self.t0 = time.time()
        self.cycles = 0
        self.by_colour = {'ryobi': 0, 'pink': 0, 'unknown': 0}
        self.lying_grabs = 0
        self.roll_aligns = 0
        self.phantom_skips = 0
        self.grab_failures = 0
        self.releases = 0            # aborted cycles (RELEASE from phase ...)
        self.scans = 0
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(LOG_DIR, f'session_{stamp}.log')
        self._fh = open(self.path, 'a', buffering=1)
        self._fh.write(f'# session started {time.ctime()}\n')

    # ---- called for every log line -------------------------------------
    _CARRY = re.compile(r'P_CARRY — (\w+) battery')

    def observe(self, msg):
        self._fh.write(msg + '\n')
        m = self._CARRY.search(msg)
        if m:
            self.by_colour[m.group(1)] = self.by_colour.get(m.group(1), 0) + 1
        elif 'PLACED — cycle complete' in msg:
            self.cycles += 1
        elif 'LYING top-down' in msg:
            self.lying_grabs += 1
        elif 'ROLL aligned' in msg:
            self.roll_aligns += 1
        elif 'SKIP phantom' in msg:
            self.phantom_skips += 1
        elif msg.startswith('RELEASE from phase') or 'RELEASE from phase' in msg:
            self.releases += 1
        elif 'SCAN — sweeping' in msg:
            self.scans += 1

    def note_grab_failure(self):
        self.grab_failures += 1

    # ---- summary --------------------------------------------------------
    def summary_dict(self):
        return {
            'uptime_s': round(time.time() - self.t0, 1),
            'cycles_completed': self.cycles,
            'sorted': self.by_colour,
            'lying_grabs': self.lying_grabs,
            'roll_aligns': self.roll_aligns,
            'grab_failures': self.grab_failures,
            'phantom_skips': self.phantom_skips,
            'aborted_cycles': self.releases,
            'scans': self.scans,
        }

    def close(self):
        s = self.summary_dict()
        self._fh.write('# SUMMARY ' + json.dumps(s) + '\n')
        self._fh.close()
        with open(self.path.replace('.log', '.json'), 'w') as f:
            json.dump(s, f, indent=2)
        return s

    def print_summary(self):
        s = self.summary_dict()
        mins = s['uptime_s'] / 60.0
        print('\n' + '=' * 52)
        print(' SESSION SUMMARY')
        print('=' * 52)
        print(f"  uptime            {mins:.1f} min")
        print(f"  cycles completed  {s['cycles_completed']}")
        print(f"  sorted  ryobi->green  {s['sorted'].get('ryobi', 0)}")
        print(f"          pink ->red    {s['sorted'].get('pink', 0)}")
        if s['sorted'].get('unknown'):
            print(f"          unknown       {s['sorted']['unknown']}")
        print(f"  lying grabs       {s['lying_grabs']}")
        print(f"  roll aligns       {s['roll_aligns']}")
        gf = s['grab_failures']
        print(f"  MISSED grabs      {gf}" + ('   <-- investigate' if gf else ''))
        print(f"  phantom skips     {s['phantom_skips']}")
        print(f"  aborted cycles    {s['aborted_cycles']}")
        print('=' * 52)
        print(f"  full log: {os.path.relpath(self.path, _HERE)}")
        print()


def install_grab_check(BaseActuator, metrics, log):
    """Wrap grip_close so a missed pick is flagged the instant it happens.

    A fixed 'near fully-closed = empty' threshold does NOT work on this
    gripper: successful grabs settle only a few raw open of fully-closed.
    So this is SELF-CALIBRATING — it measures how tight a normal grab closes
    over the session and flags only a grab that closes FAR tighter than that
    (fingers met with much less between them = probably nothing).

    Can't rewrite main.py's flow from here, but surfacing the miss loudly is
    the difference between 'why did it drop nothing in the bucket' and a
    one-line WARNING you see live and in the summary."""
    try:
        import config as cfg
        closed = cfg.GRIPPER_CLOSED_RAW
    except Exception:
        closed = None
    if closed is None:
        return
    real = BaseActuator.grip_close
    from collections import deque
    hist = deque(maxlen=8)          # recent |raw - closed| gaps
    MIN_SAMPLES = 4

    def grip_close_checked(self):
        raw = real(self)
        try:
            if raw is not None:
                gap = abs(raw - closed)      # how far it stopped OPEN of shut
                if len(hist) >= MIN_SAMPLES:
                    med = sorted(hist)[len(hist) // 2]
                    # Closed far tighter than a typical grab -> likely empty.
                    if gap < max(12, 0.4 * med):
                        metrics.note_grab_failure()
                        log(f"WARNING — grip closed unusually tight "
                            f"(gap {gap:.0f} vs typical {med:.0f}); "
                            f"likely MISSED the battery")
                    else:
                        hist.append(gap)
                else:
                    hist.append(gap)
        except Exception:
            pass
        return raw

    BaseActuator.grip_close = grip_close_checked
