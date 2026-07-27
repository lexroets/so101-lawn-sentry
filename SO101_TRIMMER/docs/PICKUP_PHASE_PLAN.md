# Phase 3 — mobile pickup: SO-101 on an autonomous mower

Goal: mower drives its normal pattern; when the camera sees poop within the
arm's reach, the mower pauses, the arm picks it up and drops it in an onboard
bin, then the mower resumes.

---

## 1. The enabling finding: TerraMow has a local API

TerraMow publish their own Home Assistant integration —
<https://github.com/TerraMow/TerraMowHA> (GPL-3.0, active, v0.3.0 Apr 2026).
It talks to the mower over **local MQTT/HTTP** (no cloud) and exposes:

- **start / pause / dock** commands
- battery + activity state
- real-time MQTT events

Requires firmware ≥ 6.6.0, app ≥ 1.6.0. There is a Developer Guide in
`docs/en/developers.md` of that repo.

**Why this matters:** `pause` is the one primitive the whole concept depends
on. You can stop the mower mid-pattern, act, and resume. That turns this from
"reverse-engineer a closed product" into "publish an MQTT message".

**What it does NOT give you:** precise pose/odometry, or "drive to (x, y)".
So you cannot command the mower to line itself up on a target. That single
limitation drives the entire architecture below.

> Worth evaluating before committing: the **Segway Navimow X3** publishes an
> *Open API for its Expansion Bay* — an officially supported mounting +
> power + data interface for third-party payloads. If you are not already
> locked to TerraMow, a platform designed to carry accessories may save you a
> lot of mechanical and electrical improvisation.

---

## 2. Architecture: stop-and-pick, not search-and-navigate

Because you can pause but not steer, v1 should be **opportunistic**:

```
mower mows normally
   └─ camera runs continuously (detector already built)
        └─ poop detected AND inside the arm's reach envelope?
             ├─ yes → MQTT pause → confirm stopped → arm picks → bin → resume
             └─ no  → log position to the map (already built) for a later pass
```

No SLAM, no global path planning, no mower steering. The arm services what
happens to pass through its workspace; everything else is logged. That is a
genuinely useful product on day one, and it degrades gracefully.

### What carries over from what you have already built

Far more than you might expect:

| Existing piece | Status on the mower |
|---|---|
| `poop_detector` / trained YOLO model | **unchanged** |
| Camera→ground **homography** | **still valid** — see below |
| Detection→metric position | unchanged (it becomes mower-relative) |
| Session logging, snapshots, CSV | unchanged |
| RYOBI dashboard + live map | becomes a *mower-relative* view, or a history feed |
| `camera_session` crash recovery | more important, not less |

**The homography survives.** Today it maps pixels → metres on a fixed lawn
rectangle. Bolted to the mower, the camera-to-ground geometry is *rigid*, so
the same homography maps pixels → metres **in front of the mower**. You stop
calling it a lawn map and start calling it a ground-plane calibration. That is
exactly the frame the arm needs. Re-run `calibrate_lawn_map.py` once with a
measured mat in front of the parked mower and you are done.

---

## 3. What is genuinely new

### 3a. Hand-eye calibration (camera → arm base)
The missing transform. A detection is currently "0.42 m ahead, 0.15 m right of
the camera"; the arm needs that in *its own* base frame to solve joint angles.

Cheap method that fits your existing tooling: place a marker, drive the arm tip
to touch it, record joint angles + the pixel/metric detection, repeat ~12 times
across the workspace, then least-squares fit the rigid transform. This is the
same "teach by demonstration" pattern you already used for the trimmer.

### 3b. Reach envelope
SO-101 reach is roughly 30–35 cm from the base. Mounted on a mower deck
(~25 cm high), the reachable patch of *ground* in front is a band perhaps
20–30 cm wide and 15–25 cm deep. **Measure it, hard-code it, and refuse
targets outside it** — attempting an unreachable pick is how you break servos.
This envelope is the single most important number in the whole phase.

### 3c. The gripper
Prior art: **Beetl** (autonomous yard poop robot) uses *clamshell jaws* — it
drives over the target and closes a claw beneath it. That is the validated
mechanism for this exact job.

For an arm-mounted version:

- **Two opposing scoop blades** that close *beneath* the object. They must
  slide a few mm into the grass to get under it — thin, bevelled leading edges.
- **One servo.** The SO-101 gripper is servo ID 6 on the same bus, so a
  clamshell is electrically a drop-in for the gripper you removed. Your
  existing `actuator.py` gripper code largely applies.
- **Compliance.** A rigid jaw pushed into uneven turf will stall the servo. A
  sprung or slightly flexible blade tolerates ground-height error — which you
  will have, because stereo depth on grass is noisy.
- **Release.** Open over a sealed onboard bin. Gravity does the rest.
- **Hygiene.** Assume the jaws get filthy. Design for tool-free removal and
  hosing, or line the bin with a bag. This is a real product concern, not a
  detail — it is what a buyer will ask about first.

Avoid suction: clogs, needs a pump, power-hungry, and wet grass makes it worse.

### 3d. Verification that the pick worked
Non-negotiable for autonomy. Cheapest reliable check: re-image the spot after
the pick and re-run the detector. Still there → retry once, then flag it and
move on. You already built exactly this pattern for the battery grabs.

---

## 4. Risks, honestly ranked

| Risk | Severity | Notes / mitigation |
|---|---|---|
| **Pick reliability on real poop** | High | Deformable, variable consistency, sticks to grass. Fresh vs. dry behave completely differently. Expect this to be the hardest single problem — budget real iteration time. |
| **Reach vs. mower position** | High | You cannot steer the mower. Many detections will simply be out of reach. Accept it; log them. |
| **Ground-height error** | Med-High | Stereo depth over grass is noisy. Compliant jaws + a touch-down/stall-detect strategy beats trying to measure height precisely. |
| **Weatherproofing** | Med | SO-101 is indoor hobby hardware. Dew alone will find every gap. Needs an enclosure, at minimum for electronics. |
| **Power & compute** | Med | Do not rely on tapping the mower's battery. Own LiFePO4 + Pi/Jetson is cleaner and doesn't void anything. |
| **Payload/torque** | Med | STS3215 ≈ 30 kg·cm. Poop 50–200 g plus a ~150 g scoop at 30 cm reach is workable but not generous. Verify shoulder torque before committing to a heavy scoop. |
| **Mower firmware changes** | Low-Med | You depend on an integration TerraMow could change. Pin firmware; the repo is GPL so you can fork. |
| **Hygiene / cross-contamination** | Med | Product-level concern. Washable jaws, sealed bin, obvious service procedure. |

---

## 5. Suggested milestones

1. **Bench the API.** Install the HA integration (or talk raw MQTT) and prove
   you can pause and resume the mower from a Python script. One afternoon. Do
   this *first* — everything else is wasted if it doesn't work.
2. **Mount + power.** Arm, camera, compute, battery on the deck. Static, not
   moving yet. Confirm the arm doesn't foul the mower or tip it.
3. **Re-calibrate ground plane** with `calibrate_lawn_map.py` against a
   measured mat in front of the parked mower.
4. **Hand-eye calibration** + measure and hard-code the reach envelope.
5. **Clamshell v1** (3D printed, one servo) — test on a *stationary* target,
   detection → pick → bin, no mower motion at all.
6. **Close the loop:** pause → pick → verify → resume, mower stationary but
   under software control.
7. **Live run** on a real lawn with test objects, then real poop.

Milestones 1–5 are all independently demoable, which matters for keeping
stakeholders warm.

---

## 6. Open questions worth settling early

- Is the TerraMow already on firmware ≥ 6.6.0, and can you get its local IP +
  MQTT password?
- Deck geometry: is there a flat, load-bearing area big enough for the arm,
  and what does it do to the mower's tip-over stability and its own obstacle
  sensors?
- Bin capacity vs. lawn size — how many picks before service?
- Does the mower's own poop-avoidance (if enabled) fight your system?
