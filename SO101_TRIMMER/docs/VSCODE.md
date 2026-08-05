# Moving from Thonny to VS Code

Nothing in the code needs changing. Every script finds its own imports through
`paths.py`, so it does not care what launched it. Two things do need attention,
and one of them is the reason scripts appear to hang.

---

## 1. The interpreter

This is the whole of it. Thonny ships its own Python with `depthai`, `opencv`,
`pyserial` and `streamlit` already installed. VS Code uses whichever
interpreter you pick, and your shell prompt says `(base)` — a conda
environment that almost certainly has none of them. Point VS Code at the wrong
one and everything fails at the first `import depthai`.

**Quickest — borrow Thonny's Python.** Works immediately, nothing to install:

`Cmd+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** →

```
/Applications/Thonny.app/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3.14
```

To confirm that is the right path, run this in Thonny's shell:

```python
import sys; print(sys.executable)
```

**Better — your own environment.** Thonny's Python is 3.14, and
**`onnxruntime` has no 3.14 wheel**, which is why the detector runs on the
`cv2.dnn` fallback. That fallback is the slowest thing in the scan loop. A
3.11 or 3.12 environment gets you the fast path:

```bash
conda create -n sentry python=3.12 -y
conda activate sentry
cd ~/so101_autonomy_architecture_disabled/SO101_TRIMMER
pip install -r requirements.txt
```

Then pick `sentry` in the interpreter list. Check it took:

```python
import depthai, cv2, serial, streamlit, onnxruntime   # all should import
```

---

## 2. `input()` — the one that will confuse you

`calibrate.py` asks for the lawn dimensions. `launch.py` waits on *"press
ENTER to close"*. `calibrate_reach.py` waits for you to capture each point.

**VS Code's default debug console cannot accept typed input.** Those scripts do
not error — they sit there forever with a cursor that does nothing. Thonny's
shell handled input transparently, so this never came up.

The fix is one setting, and it is already in `.vscode/launch.json`:

```json
"console": "integratedTerminal"
```

Every configuration in there has it. If you add your own, copy that line.

Running with **Ctrl+F5** or from the terminal (`python launch.py`) is also
fine — it is only the debug console that has this problem.

---

## 3. What is already set up for you

`.vscode/launch.json` — press **F5** and pick one:

| | |
| --- | --- |
| Run the demo | `launch.py` |
| Set the system up | `calibrate.py` |
| Pre-demo check | `tools/demo_check.py` |
| Trimmer: reach an object | `trimmer/reach_object.py` |
| Trimmer: re-teach the reach table | `trimmer/calibrate/calibrate_reach.py` |
| Run the file I'm looking at | whatever is open |

All of them set the working directory to the project root, which matters:
`launch.py` starts Streamlit from there and `.streamlit/` lives there.

`.vscode/settings.json` also lists the code folders under
`python.analysis.extraPaths`. That is **only** so the editor stops underlining
`import poop_config` — the code resolves those itself at runtime and would work
without it.

---

## 4. Things that behave differently

**Open the `SO101_TRIMMER` folder**, not the parent. Git still works — VS Code
finds the repository above it.

**OpenCV windows.** `cv2.imshow` windows open behind the editor on macOS.
Click the window before pressing keys — keys only reach the focused window, and
that catches people out with `reach_object.py` and `calibrate.py`.

**Stopping a run.** Thonny's red button killed everything. In VS Code use the
stop button on the debug toolbar, or `Ctrl+C` in the terminal. If a run is
killed hard, the OAK-D can stay claimed — unplug and replug it, or wait a few
seconds. `launch.py` already clears ports 8501 and 8799 itself.

**The serial port.** Only one process can hold it. If you get *"could not open
/dev/cu.usbmodem..."*, something is still running — check for stray terminals,
and make sure Thonny is fully quit rather than just in the background.
