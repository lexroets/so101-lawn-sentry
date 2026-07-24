"""Unattended production runner — keeps the sorter alive.

Runs main_ml.py in a child process and restarts it if it CRASHES (camera
hiccup, serial glitch, dropped USB). A clean quit (you press q) stops for
good, and a fatal config problem (preflight NO-GO) stops immediately rather
than looping — because restarting won't fix a missing calibration.

    python run_production.py

Each attempt is a fresh process, so there is no state to leak between runs:
the camera is fully released and re-acquired, and nothing gets double-patched.

Exit codes from main_ml.py:
    0  clean quit (operator pressed q)      -> stop
    2  fatal config / preflight NO-GO       -> stop, show the fix
    *  crash                                -> restart (with backoff + cap)
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(_HERE, 'main_ml.py')

BACKOFF_S = 4.0            # wait this long after a crash before restarting
CRASH_WINDOW_S = 120.0     # if too many crashes happen within this window...
CRASH_LIMIT = 5            # ...give up (something is persistently wrong)


def main():
    crashes = []           # timestamps of recent crashes
    attempt = 0
    while True:
        attempt += 1
        print(f'\n===== launch #{attempt}  {time.ctime()} =====')
        started = time.time()
        try:
            rc = subprocess.call([sys.executable, TARGET], cwd=_HERE)
        except KeyboardInterrupt:
            print('\nrun_production: Ctrl-C — stopping.')
            return

        if rc == 0:
            print('run_production: clean exit — stopping.')
            return
        if rc == 2:
            print('run_production: fatal config (preflight NO-GO) — stopping. '
                  'Fix the FAIL rows above and restart.')
            return

        # --- it crashed ---------------------------------------------------
        now = time.time()
        ran_for = now - started
        crashes = [t for t in crashes if now - t <= CRASH_WINDOW_S]
        crashes.append(now)
        print(f'run_production: crash (exit {rc}) after {ran_for:.0f}s. '
              f'{len(crashes)} crash(es) in the last '
              f'{int(CRASH_WINDOW_S)}s.')
        if len(crashes) >= CRASH_LIMIT:
            print(f'run_production: {CRASH_LIMIT} crashes in a row — '
                  'giving up. Check the camera/USB and the latest '
                  'session_logs/ file.')
            return
        print(f'run_production: restarting in {BACKOFF_S:.0f}s...')
        try:
            time.sleep(BACKOFF_S)
        except KeyboardInterrupt:
            print('\nrun_production: Ctrl-C — stopping.')
            return


if __name__ == '__main__':
    main()
