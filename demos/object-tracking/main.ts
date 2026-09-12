import { FilesetResolver, ObjectDetector } from "@mediapipe/tasks-vision";
import { ByteTracker, type Det } from "./tracker";

const WASM = "./mp/wasm";
const MODEL = "./mp/model/efficientdet.tflite";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $("cam") as HTMLVideoElement;
const canvas = $("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const statusEl = $("status");
const startBtn = $("start") as HTMLButtonElement;
const fpsEl = $("fps");
const countEl = $("count");

let detector: ObjectDetector | null = null;
let running = false;
const tracker = new ByteTracker({ highThresh: 0.45, lowThresh: 0.15, iouMatch: 0.25, minHits: 3, maxAge: 30, trail: 26 });
let lastT = 0, fps = 0, lastVideoTime = -1;

async function initModel() {
  statusEl.textContent = "loading detector…";
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  detector = await ObjectDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    scoreThreshold: 0.12,      // low, on purpose: the tracker uses the low-conf tail
    maxResults: 25,
  });
  statusEl.textContent = "detector ready — click Enable camera.";
  startBtn.disabled = false;
}

async function startCamera() {
  startBtn.disabled = true;
  statusEl.textContent = "requesting camera…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 960, height: 720 }, audio: false });
    video.srcObject = stream;
    await video.play();
    running = true;
    statusEl.textContent = "tracking — point the camera at people or objects";
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

function draw(tracks: ReturnType<ByteTracker["step"]>) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = Math.max(2, W / 480);
  ctx.font = `${Math.round(W / 42)}px "IBM Plex Mono", monospace`;
  ctx.textBaseline = "top";
  for (const t of tracks) {
    const x = t.x * W, y = t.y * H, w = t.w * W, h = t.h * H;
    // motion trail
    if (t.trail.length > 1) {
      ctx.strokeStyle = t.color; ctx.globalAlpha = 0.35; ctx.beginPath();
      t.trail.forEach((p, i) => { const px = p.x * W, py = p.y * H; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = t.color; ctx.strokeRect(x, y, w, h);
    // label chip: "#id label 0.87"
    const tag = `#${t.id} ${t.label} ${t.score.toFixed(2)}`;
    const tw = ctx.measureText(tag).width + 10, th = W / 30;
    ctx.fillStyle = t.color; ctx.fillRect(x, y - th, tw, th);
    ctx.fillStyle = "#0b0b0c"; ctx.fillText(tag, x + 5, y - th + th * 0.18);
  }
  fpsEl.textContent = `${fps.toFixed(0)} fps`;
  countEl.textContent = `${tracker.count} tracked`;
}

function loop(t: number) {
  if (!running || !detector) return;
  sizeCanvas();
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = detector.detectForVideo(video, t);
    const W = video.videoWidth || 960, H = video.videoHeight || 720;
    const dets: Det[] = (res.detections || []).map((d) => {
      const b = d.boundingBox!; const c = d.categories[0];
      return { x: b.originX / W, y: b.originY / H, w: b.width / W, h: b.height / H, score: c.score, cls: c.index, label: c.categoryName || "object" };
    });
    draw(tracker.step(dets));
    if (lastT) { const f = 1000 / (t - lastT); fps = fps ? fps * 0.9 + f * 0.1 : f; }
    lastT = t;
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
initModel().catch((e) => { statusEl.textContent = "⚠ " + (e?.message || e); console.error(e); });
