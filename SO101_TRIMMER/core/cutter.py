"""The cutting motor, and the rules about when it is allowed to spin.

The arm can position the head perfectly and the motor is a separate thing
entirely - nothing in this project drove it. That gap is worth closing
carefully rather than quickly, because a spinning cutter is the one part of
this machine that can hurt someone, and the failure mode that matters is not
"it did not start" but "it did not stop".

So the design is: it is OFF unless something is actively keeping it on.

    cutter.arm()                 permit spinning at all (once, at startup)
    cutter.on()  / cutter.off()  request / stop
    cutter.keepalive()           must be called or it stops on its own
    cutter.stop()                unconditional, safe from anywhere

Everything that can go wrong stops it:

    * a watchdog - if the control loop stops calling keepalive() for
      KEEPALIVE_TIMEOUT_S, the motor stops. A crashed loop, a wedged camera or
      a closed window therefore cannot leave it running.
    * MAX_RUN_S - a hard ceiling on one continuous run, whatever anyone asks.
    * atexit and the context manager - normal exit, exception or Ctrl-C all
      run the stop path.
    * stop() never raises. A stop that can fail is not a stop.

WIRING - this is the part only you can fill in
----------------------------------------------
Set BACKEND below to match how the motor is actually driven. Until then it
runs in 'sim' and prints what it would have done, so the rest of the system
can be developed and demonstrated safely.

    'sim'      no hardware; prints. THE DEFAULT.
    'gpio'     a relay or ESC on a Raspberry Pi pin      -> set GPIO_PIN
    'serial'   an Arduino or driver board on a port      -> set SERIAL_PORT,
               sends b'1' to run and b'0' to stop
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
import atexit
import threading
import time

# ---- wiring ----------------------------------------------------------------
BACKEND = 'sim'             # 'sim' | 'gpio' | 'serial'
GPIO_PIN = 18               # BCM numbering, for BACKEND='gpio'
SERIAL_PORT = None          # e.g. '/dev/cu.usbmodem14101', for BACKEND='serial'
SERIAL_BAUD = 115200

# ---- safety ----------------------------------------------------------------
# The loop must check in at least this often or the motor stops. Generous
# enough not to trip on a slow frame, short enough that a wedged loop is not
# left spinning.
KEEPALIVE_TIMEOUT_S = 1.0
# Nothing runs longer than this in one go, whatever is asked of it.
MAX_RUN_S = 30.0
# Let it come up to speed before the arm treats it as cutting.
SPIN_UP_S = 0.6


class Cutter:
    """Fail-off motor control. Safe to call from any thread."""

    def __init__(self, backend=BACKEND):
        self.backend = backend
        self._lock = threading.RLock()
        self._armed = False
        self._running = False
        self._started = 0.0
        self._last_ping = 0.0
        self._io = None
        self._stop_evt = threading.Event()
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()
        atexit.register(self.stop)

    # -- hardware ------------------------------------------------------------
    def _open(self):
        if self._io is not None or self.backend == 'sim':
            return
        if self.backend == 'gpio':
            import RPi.GPIO as GPIO          # noqa: N814
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
            self._io = GPIO
        elif self.backend == 'serial':
            import serial
            if not SERIAL_PORT:
                raise RuntimeError('BACKEND is "serial" but SERIAL_PORT is unset')
            self._io = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2)
            time.sleep(2.0)                  # boards reset when the port opens

    def _drive(self, on):
        if self.backend == 'sim':
            print(f'[cutter] (sim) motor {"ON" if on else "OFF"}')
            return
        if self._io is None:
            return
        if self.backend == 'gpio':
            self._io.output(GPIO_PIN, self._io.HIGH if on else self._io.LOW)
        elif self.backend == 'serial':
            self._io.write(b'1' if on else b'0')
            self._io.flush()

    # -- control -------------------------------------------------------------
    def arm(self):
        """Permit the motor to run at all. Nothing spins before this."""
        with self._lock:
            try:
                self._open()
            except Exception as e:
                print(f'[cutter] backend "{self.backend}" unavailable ({e}) '
                      f'- staying disarmed')
                return False
            self._armed = True
            print(f'[cutter] armed, backend={self.backend}')
            return True

    def on(self):
        """Request the motor. Returns False if it is not allowed to run."""
        with self._lock:
            if not self._armed:
                return False
            self._last_ping = time.monotonic()
            if self._running:
                return True
            self._drive(True)
            self._running = True
            self._started = time.monotonic()
            return True

    def keepalive(self):
        """Tell the watchdog the control loop is still alive and still wants it."""
        self._last_ping = time.monotonic()

    def off(self):
        with self._lock:
            if self._running:
                self._drive(False)
                self._running = False

    def stop(self):
        """Unconditional. Never raises - a stop that can fail is not a stop."""
        try:
            with self._lock:
                if self._running:
                    try:
                        self._drive(False)
                    except Exception:
                        pass
                    self._running = False
                self._armed = False
        except Exception:
            pass

    @property
    def running(self):
        return self._running

    def at_speed(self):
        """True once it has had time to spin up."""
        return self._running and (time.monotonic() - self._started) >= SPIN_UP_S

    def status(self):
        if not self._armed:
            return 'disarmed'
        if not self._running:
            return 'idle'
        return f'RUNNING {time.monotonic() - self._started:.1f}s'

    # -- the thing that makes it safe ---------------------------------------
    def _watchdog(self):
        while not self._stop_evt.wait(0.1):
            if not self._running:
                continue
            now = time.monotonic()
            if now - self._last_ping > KEEPALIVE_TIMEOUT_S:
                print('[cutter] no keepalive - stopping')
                self.off()
            elif now - self._started > MAX_RUN_S:
                print(f'[cutter] hit the {MAX_RUN_S:.0f}s run limit - stopping')
                self.off()

    # -- context manager: exceptions and Ctrl-C stop it too -----------------
    def __enter__(self):
        self.arm()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def close(self):
        self.stop()
        self._stop_evt.set()
        if self._io is not None and self.backend == 'serial':
            try:
                self._io.close()
            except Exception:
                pass
