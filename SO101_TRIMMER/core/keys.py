"""One press, one action - including when the loop is busy talking to servos.

Two separate things go wrong with `cv2.waitKey`, and fixing only the first one
is why the buttons still needed spamming.

FIRST: it reports which key is DOWN, not when it was pressed. Act on the value
and you act again on every poll while it is held - one tap of ENTER toggling
auto-mode five times. Edge detection fixes that, and a time debounce does not:
too short and a held key still repeats, too long and a deliberate double-tap is
swallowed, and one shared timer makes SPACE block ENTER.

SECOND, and the one that actually mattered: `waitKey` is not a keyboard read,
it is the GUI event loop. Keys only reach the program while it is running.
A loop that spends 300 ms on blocking servo reads and then calls waitKey(1)
gives that event loop about one percent of the time, so almost every press
lands in the dead gap and is never seen. The keyboard felt broken because it
effectively was.

So the event loop is PUMPED often - `pump()` is cheap and safe to call
anywhere, including in the middle of slow work - and presses are LATCHED into a
queue rather than read from whatever waitKey happens to return at the bottom of
the loop. Pumping ten times and handling once no longer loses nine of them.

    kb.pump()                 # sprinkle through the loop, especially near
                              # anything slow - it costs a millisecond
    if kb.hit(ord(' ')): ...  # consumes one press
    if kb.tap(ord('o')): ...  # consumes a press OR an auto-repeat
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
import time

import cv2

NONE = 255
REPEAT_DELAY_S = 0.45      # hold this long before a trim key starts repeating
REPEAT_EVERY_S = 0.12
# A press nobody asked about is dropped after this, so a stray key cannot fire
# something surprising much later.
STALE_S = 1.5


class Keys:
    """Edge-detected, latched keyboard for an OpenCV window."""

    def __init__(self, repeat_delay=REPEAT_DELAY_S, repeat_every=REPEAT_EVERY_S):
        self._prev = NONE
        self._down_at = 0.0
        self._last_repeat = 0.0
        self.repeat_delay = repeat_delay
        self.repeat_every = repeat_every
        self._pending = []        # [(key, is_repeat, when)]
        self.key = NONE

    # -- pumping -------------------------------------------------------------
    def pump(self, wait_ms=1):
        """Run the GUI event loop once and latch anything new.

        Call it as often as you like. The queue is what makes that safe: an
        earlier pump cannot steal a press from a later hit().
        """
        k = cv2.waitKey(wait_ms) & 255
        now = time.monotonic()
        if k != NONE:
            if k != self._prev:
                self._pending.append((k, False, now))
                self._down_at = now
                self._last_repeat = now
            elif (now - self._down_at >= self.repeat_delay
                  and now - self._last_repeat >= self.repeat_every):
                self._pending.append((k, True, now))
                self._last_repeat = now
        self._prev = k
        self.key = k
        # forget anything nobody claimed
        self._pending = [e for e in self._pending if now - e[2] < STALE_S]
        return self

    # kept so older call sites still work
    def poll(self, wait_ms=1):
        return self.pump(wait_ms)

    # -- consuming -----------------------------------------------------------
    def _take(self, codes, allow_repeat):
        for i, (k, is_rep, _t) in enumerate(self._pending):
            if k in codes and (allow_repeat or not is_rep):
                del self._pending[i]
                return True
        return False

    def hit(self, *codes):
        """True once per physical press. For anything with a consequence."""
        return self._take(codes, allow_repeat=False)

    def tap(self, *codes):
        """True on press, and again while held. For nudges and trims."""
        return self._take(codes, allow_repeat=True)

    def clear(self):
        self._pending.clear()

    @property
    def pending(self):
        return [k for k, _r, _t in self._pending]

    @property
    def any(self):
        return bool(self._pending)
