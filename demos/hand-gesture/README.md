# Real-time Hand Gesture Recognition

In-browser hand tracking with **MediaPipe Tasks Vision** (`HandLandmarker`).
21 3D landmarks per hand are detected at video frame-rate on the GPU (WebAssembly),
and a small geometric classifier (`main.ts`) reads finger states into gestures:
Fist, Open Palm, Peace, Thumbs Up, Pointing, Call Me, Love, Rock On, OK.

- **No server, no upload** — camera frames never leave the browser.
- Model + wasm are **self-hosted** (`mp/`) so the demo loads same-origin with no CDN dependency.

`main.ts` is the full source; the page is a static Vite build.
