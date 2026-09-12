# Real-time Multi-Object Tracking (ByteTrack-style)

In-browser detection + tracking. A **MediaPipe** EfficientDet-Lite detector finds
objects each frame; a from-scratch **ByteTrack-style tracker** (`tracker.ts`,
dependency-free) links them into persistent tracks with stable IDs, a
constant-velocity motion model, and motion trails.

The tracker implements ByteTrack's two-stage association (Zhang et al., ECCV 2022):
high-confidence detections are matched first, then leftover tracks are recovered by
matching the **low-confidence** detections — which keeps IDs alive through motion
blur and brief occlusion instead of dropping them.

- **No upload** — camera frames are processed entirely client-side.
- Model + wasm are self-hosted (`mp/`) so it loads same-origin.
- `tracker.ts` is the core algorithm; `main.ts` wires detector → tracker → canvas.
