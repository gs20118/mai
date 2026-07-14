# Practice session capture checklist

Everything in `configs/arena.yaml` under `aruco:` is a **placeholder**. Until these
are measured, the pipeline is running on guesses. This list exists so one visit to
the arena gets everything, because a second visit may not be available.

Print it. Tick it off on-site.

---

## 1. Measure the ArUco markers — blocks everything else

The homography is anchored on four corner markers, and the whole project's
coordinate system rides on them.

- [ ] **Dictionary.** Shoot one frame of the arena, then run
      `python -m mai.cli.scan_aruco --source <frame>`. Paste the reported
      dictionary into `arena.yaml`.
- [ ] **Marker size.** Measure the edge of the **black square** (not the white
      card) in cm → `marker_size_cm`.
- [ ] **Marker centres.** Measure each marker's centre in arena cm. Origin is the
      **top-left corner of the arena**, +x right, +y down. Record which physical
      corner carries which ID.
- [ ] **Marker rotation.** Is any marker taped down rotated? ArUco decodes
      orientation, so a rotated marker whose `rotation_deg` we don't record will
      skew the homography while still reporting a plausible-looking fit.
- [ ] **White margin?** Photograph a marker close up. ArUco finds markers as a dark
      quad on a light background. If the markers are printed edge-to-edge with no
      white quiet zone, detection will be unreliable and we need to know now.

> **Known risk.** To frame the whole 500×400cm arena the drone must fly high enough
> that a 10cm marker spans only **~40px** of a 4K frame — right at ArUco's
> reliability floor. `mai.aruco.build_detector` is already tuned for small markers,
> but if the real markers are small, **verify all four detect at your planned
> altitude before trusting the plan.** Losing even one marker means the frame
> cannot be registered at all.

## 2. Confirm the arena layout

`arena.yaml` encodes five 80cm bands: **FA / TW-A / RW / TW-B / FA**.

- [ ] Confirm that band order against the real arena.
- [ ] Confirm **which physical row holds FA-01..03** and which holds FA-04..06.
- [ ] Confirm FA-02 and FA-05 are the 180cm-wide middle buildings of each row.
- [ ] Photograph the zone ID labels if the arena has any printed on it.

The check that proves all of this at once: run `build_topview` and look at
`grid_on_source.jpg`. The drawn grid must land on the real runway and taxiway
edges. If it does, the marker map, the distortion profile, the homography and the
band layout are all correct simultaneously.

## 3. Time the transfer chain — the single most important number

The mission is 180 seconds from takeoff to landing, and **JSON received after
landing does not count**. Inference is not the bottleneck; moving files off the
drone is. A 60-second 4K clip is roughly 700MB–1GB and could take longer to
transfer than the entire mission window.

Stopwatch, end to end, from shutter press to file readable on the laptop:

- [ ] One 4K **photo**: ______ seconds
- [ ] A 10-second 4K **video clip**: ______ seconds
- [ ] Does the transfer program deliver files **incrementally**, or only after the
      drone lands? (If incremental, we can infer on photo #1 while photo #5 is
      still being taken.)

This number decides whether the flight plan is stills-at-waypoints or continuous
video. Bring it back above all else.

## 4. Shoot the footage

For each capture, **write down which mode it was shot in** — photo and video have
different intrinsics and need separate `camera.yaml` profiles.

- [ ] **Whole-arena hover**, all four markers in frame, at 2–3 altitudes. This is
      the top-view the geometry pipeline consumes.
- [ ] Same, in **both photo mode and video mode**.
- [ ] A frame containing a **long straight edge** (a runway edge running the full
      500cm) — this is what the distortion tuner's straightness probe needs.
- [ ] **Lower passes** over the runway and taxiways. See the resolution note below.
- [ ] **Facility close-ups**: fly the facility rows lower, gimbal tilted ~45°, to
      see building *facades*. A nadir shot shows roofs, and damage is on the sides.

## 5. Build the training set while you have the arena

You control object placement here, and that is worth more than any modelling
trick: **record the ground-truth zone of every object you place.** With the
homography working, known world positions project into every frame automatically,
which auto-generates bounding boxes and turns days of labelling into an afternoon.

- [ ] **Facilities — the cheapest 18 points in the competition.** Positions never
      change, only state. Photograph all 6 buildings in all 3 states
      (`normal` / `destory` / `fire`) from the altitude and angle you will actually
      fly. Do this first if time is short.
- [ ] **Craters**: all 3 sizes, in a range of RW and TW zones. Record zone + size.
- [ ] **UXO**: all 3 types (`misile` / `dumb` / `cluster`), varied positions.
      Record zone + type.
- [ ] Lay out **all objects at once** in known positions and fly a long capture —
      every frame then yields labels for every object.
- [ ] Shoot under **different lighting** if at all possible. The rules warn that
      venue lighting may change on the day.

## 6. Resolution reality check

From the synthetic model, a single whole-arena shot at ~300cm gives about
**2.3 mm/px**, which puts targets at roughly:

| target | size | pixels | verdict |
|---|---|---|---|
| crater (big / medium / small) | 179 / 159 / 102mm | 74 / 66 / 43px | comfortable |
| UXO missile | 50mm | ~21px | OK |
| UXO shell (`dumb`) | 44mm | ~18px | marginal |
| **UXO cluster** | **28mm** | **~12px** | **marginal** |

Note *why* this is worse than it looks: the arena is 5:4 but the sensor is 16:9, so
framing the whole arena is limited by its 400cm **depth** against 2160px of frame
height — not by width. That costs ~30% of the resolution one would naively expect.

So one whole-arena shot probably cannot carry the 18-point UXO mission on its own.

- [ ] Fly a **lower pass over the runway/taxiway strip** and check whether the
      cluster munition becomes clearly resolvable. Note the altitude that works.
- [ ] Check whether all four markers are still visible at that lower altitude. If
      not, that pass cannot be registered from markers alone and we will need
      feature matching against the top-view instead.

## 7. Ask the organizers

Wrong guesses here cost double-digit points:

- [ ] Exact JSON schema for `facility_status`, `uxo_detect`, `uxo_count`,
      `crater_count`, `report`. Only three examples appear in TASK.md. **How are
      facilities keyed — by FA zone ID, or by building name?**
- [ ] Dashboard protocol: TCP socket, HTTP POST, or file drop? Host, port, framing?
- [ ] Duplicate JSON: does **last received win**, or first?
- [ ] Any **penalty for false positives**? (Decides whether we always fill all 5
      crater slots and all 6 UXO slots, which is otherwise free expected value.)
- [ ] Crater **partial credit**: zone right but size wrong — 0, or partial?
- [ ] Is capturing the **live video feed** allowed, or must everything go through
      the provided transfer program? If live capture is allowed, the transfer
      bottleneck disappears entirely.
