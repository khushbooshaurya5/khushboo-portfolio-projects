import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// Self-hosted: wasm + model ship with the demo, so it loads same-origin (no CDN dependency).
const WASM = "./mp/wasm";
const MODEL = "./mp/model/hand_landmarker.task";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $("cam") as HTMLVideoElement;
const canvas = $("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const statusEl = $("status");
const gEmoji = $("gEmoji");
const gName = $("gName");
const gHint = $("gHint");
const fpsEl = $("fps");
const startBtn = $("start") as HTMLButtonElement;

type Pt = { x: number; y: number; z: number };

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],           // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // index
  [5, 9], [9, 10], [10, 11], [11, 12],      // middle
  [9, 13], [13, 14], [14, 15], [15, 16],    // ring
  [13, 17], [17, 18], [18, 19], [19, 20],   // pinky
  [0, 17],                                  // palm base
];

let landmarker: HandLandmarker | null = null;
let running = false;
let lastT = 0;
let fpsSmooth = 0;

async function initModel() {
  statusEl.textContent = "loading hand-tracking model…";
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
  statusEl.textContent = "model ready — click Enable camera.";
  startBtn.disabled = false;
}

async function startCamera() {
  startBtn.disabled = true;
  statusEl.textContent = "requesting camera…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 960, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    statusEl.textContent = "tracking — show your hand ✋";
    requestAnimationFrame(loop);
  } catch (e: any) {
    statusEl.textContent = "⚠ camera blocked — allow access and reload. (" + (e?.name || e) + ")";
    startBtn.disabled = false;
  }
}

function sizeCanvas() {
  const w = video.videoWidth || 960, h = video.videoHeight || 720;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

// ---- gesture classification from 21 landmarks ----
function d(a: Pt, b: Pt) { return Math.hypot(a.x - b.x, a.y - b.y); }
function proj(p: Pt, o: Pt, dir: Pt) { return (p.x - o.x) * dir.x + (p.y - o.y) * dir.y; }
// interior angle at joint b formed by a-b-c (radians)
function angle(a: Pt, b: Pt, c: Pt) {
  const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y;
  const dot = v1x * v2x + v1y * v2y;
  const m = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y) || 1e-6;
  return Math.acos(Math.max(-1, Math.min(1, dot / m)));
}

function fingerStates(lm: Pt[]): boolean[] {
  const wrist = lm[0], midMcp = lm[9];
  let dx = midMcp.x - wrist.x, dy = midMcp.y - wrist.y;
  const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;          // palm "up" axis
  const dir = { x: dx, y: dy, z: 0 };
  const scale = d(wrist, midMcp) || 1;
  // index..pinky: tip projects further along palm axis than its PIP joint
  const fingers = [[8, 6], [12, 10], [16, 14], [20, 18]].map(
    ([tip, pip]) => proj(lm[tip], wrist, dir) > proj(lm[pip], wrist, dir) + scale * 0.15
  );
  // thumb: straight at the IP joint (extended thumbs are near-collinear 2-3-4; a curled one bends sharply)
  const thumb = angle(lm[2], lm[3], lm[4]) > 2.35;
  return [thumb, ...fingers];
}

function classify(lm: Pt[]): { emoji: string; name: string } {
  const [t, i, m, r, p] = fingerStates(lm);
  const scale = d(lm[0], lm[9]) || 1;
  const key = `${+t}${+i}${+m}${+r}${+p}`;
  // OK sign: thumb + index tips pinched, other three out
  if (d(lm[4], lm[8]) < scale * 0.35 && m && r && p) return { emoji: "👌", name: "OK" };
  const table: Record<string, { emoji: string; name: string }> = {
    "00000": { emoji: "✊", name: "Fist" },
    "11111": { emoji: "✋", name: "Open Palm" },
    "01111": { emoji: "🖐", name: "Four" },
    "01000": { emoji: "☝️", name: "Pointing" },
    "01100": { emoji: "✌️", name: "Peace" },
    "10000": { emoji: "👍", name: "Thumbs Up" },
    "01110": { emoji: "🖖", name: "Three" },
    "11100": { emoji: "3️⃣", name: "Three" },
    "10001": { emoji: "🤙", name: "Call Me" },
    "01001": { emoji: "🤘", name: "Rock On" },
    "11001": { emoji: "🤟", name: "Love" },
    "11000": { emoji: "👆", name: "Gun / L" },
  };
  if (table[key]) return table[key];
  const count = [t, i, m, r, p].filter(Boolean).length;
  return { emoji: "🖐", name: `${count} finger${count === 1 ? "" : "s"} up` };
}

// ---- drawing ----
function draw(hands: Pt[][], handed: string[]) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width, 0); ctx.scale(-1, 1);   // mirror to match the flipped video
  for (const lm of hands) {
    const pts = lm.map((q) => ({ x: q.x * canvas.width, y: q.y * canvas.height }));
    ctx.lineWidth = 5; ctx.lineCap = "round"; ctx.shadowColor = "#D9502E"; ctx.shadowBlur = 12;
    ctx.strokeStyle = "rgba(217,80,46,.9)";
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
    }
    ctx.shadowBlur = 0;
    for (let k = 0; k < pts.length; k++) {
      const tip = k === 4 || k === 8 || k === 12 || k === 16 || k === 20;
      ctx.beginPath(); ctx.arc(pts[k].x, pts[k].y, tip ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = tip ? "#ECE7DD" : "#f0d24b"; ctx.fill();
    }
  }
  ctx.restore();
  fpsEl.textContent = hands.length
    ? `${hands.length} hand${hands.length > 1 ? "s" : ""} · ${fpsSmooth.toFixed(0)} fps`
    : `${fpsSmooth.toFixed(0)} fps`;
}

function loop(t: number) {
  if (!running || !landmarker) return;
  sizeCanvas();
  if (video.readyState >= 2) {
    const res = landmarker.detectForVideo(video, t);
    const hands = (res.landmarks || []) as Pt[][];
    const handed = (res.handednesses || []).map((h) => h[0]?.categoryName || "");
    draw(hands, handed);
    if (hands.length) {
      const g = classify(hands[0]);
      gEmoji.textContent = g.emoji; gName.textContent = g.name;
      gHint.textContent = hands.length > 1 ? "(reading your first hand)" : "";
    } else {
      gEmoji.textContent = "🫥"; gName.textContent = "no hand"; gHint.textContent = "show your hand to the camera";
    }
    if (lastT) { const fps = 1000 / (t - lastT); fpsSmooth = fpsSmooth ? fpsSmooth * 0.9 + fps * 0.1 : fps; }
    lastT = t;
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
initModel().catch((e) => { statusEl.textContent = "⚠ " + (e?.message || e); console.error(e); });
