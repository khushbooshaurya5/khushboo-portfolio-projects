/**
 * ByteTrack-style multi-object tracker (browser, dependency-free).
 *
 * Turns per-frame detections into persistent tracks with stable IDs. Implements
 * the key idea from ByteTrack (Zhang et al., ECCV 2022): a *two-stage*
 * association that first matches high-confidence detections, then recovers
 * existing tracks by matching the leftover LOW-confidence detections — which is
 * what keeps IDs alive through occlusion and motion blur, instead of dropping
 * them the moment the detector's score dips.
 *
 * Association is greedy-by-IoU with a gate; each track carries a constant-
 * velocity motion model so a box is predicted forward before matching.
 */

export interface Det { x: number; y: number; w: number; h: number; score: number; cls: number; label: string; }

export interface Track {
  id: number;
  x: number; y: number; w: number; h: number;   // bbox, top-left + size (normalised 0..1)
  vx: number; vy: number;                        // centre velocity per frame
  score: number; cls: number; label: string;
  hits: number;                                  // total matched frames
  age: number;                                   // frames since birth
  since: number;                                 // frames since last match
  confirmed: boolean;
  color: string;
  trail: { x: number; y: number }[];             // recent centre points
}

const cx = (b: { x: number; w: number }) => b.x + b.w / 2;
const cy = (b: { y: number; h: number }) => b.y + b.h / 2;

function iou(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const iw = Math.max(0, x2 - x1), ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const uni = a.w * a.h + b.w * b.h - inter;
  return uni <= 0 ? 0 : inter / uni;
}

// hsl colour cycled by track id, so each id gets a stable, distinct hue
function colorFor(id: number) { return `hsl(${(id * 47) % 360} 85% 55%)`; }

// greedy IoU matching between `tracks` (predicted) and `dets`, gated by iouMin.
// returns matched pairs plus the indices left unmatched on each side.
function match(tracks: Track[], dets: Det[], iouMin: number) {
  const pairs: { t: number; d: number; iou: number }[] = [];
  for (let i = 0; i < tracks.length; i++)
    for (let j = 0; j < dets.length; j++) {
      const v = iou(tracks[i], dets[j]);
      if (v >= iouMin) pairs.push({ t: i, d: j, iou: v });
    }
  pairs.sort((p, q) => q.iou - p.iou);
  const tUsed = new Set<number>(), dUsed = new Set<number>();
  const matched: { t: number; d: number }[] = [];
  for (const p of pairs) {
    if (tUsed.has(p.t) || dUsed.has(p.d)) continue;
    tUsed.add(p.t); dUsed.add(p.d); matched.push({ t: p.t, d: p.d });
  }
  const tFree = tracks.map((_, i) => i).filter((i) => !tUsed.has(i));
  const dFree = dets.map((_, j) => j).filter((j) => !dUsed.has(j));
  return { matched, tFree, dFree };
}

export interface TrackerOpts {
  highThresh?: number;   // detections >= this are "high confidence"
  lowThresh?: number;    // detections in [low, high) join stage-2 recovery
  iouMatch?: number;     // IoU gate for association
  minHits?: number;      // frames before a track is confirmed / drawn
  maxAge?: number;       // frames a track survives unmatched before deletion
  trail?: number;        // trail length
}

export class ByteTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private o: Required<TrackerOpts>;
  constructor(opts: TrackerOpts = {}) {
    this.o = {
      highThresh: opts.highThresh ?? 0.45,
      lowThresh: opts.lowThresh ?? 0.15,
      iouMatch: opts.iouMatch ?? 0.25,
      minHits: opts.minHits ?? 3,
      maxAge: opts.maxAge ?? 30,
      trail: opts.trail ?? 24,
    };
  }

  private predict(t: Track) {
    t.x += t.vx; t.y += t.vy; t.age++; t.since++;
  }
  private update(t: Track, d: Det) {
    const pcx = cx(t), pcy = cy(t);
    // EMA-smooth the box, then re-derive velocity from the centre motion
    const a = 0.5;
    t.x = a * d.x + (1 - a) * t.x; t.y = a * d.y + (1 - a) * t.y;
    t.w = a * d.w + (1 - a) * t.w; t.h = a * d.h + (1 - a) * t.h;
    t.vx = cx(t) - pcx; t.vy = cy(t) - pcy;
    t.score = d.score; t.cls = d.cls; t.label = d.label;
    t.hits++; t.since = 0;
    if (t.hits >= this.o.minHits) t.confirmed = true;
    t.trail.push({ x: cx(t), y: cy(t) });
    if (t.trail.length > this.o.trail) t.trail.shift();
  }
  private spawn(d: Det) {
    const t: Track = {
      id: this.nextId++, x: d.x, y: d.y, w: d.w, h: d.h, vx: 0, vy: 0,
      score: d.score, cls: d.cls, label: d.label, hits: 1, age: 1, since: 0,
      confirmed: this.o.minHits <= 1, color: colorFor(this.nextId), trail: [{ x: cx(d), y: cy(d) }],
    };
    t.color = colorFor(t.id);
    this.tracks.push(t);
  }

  /** Advance one frame with the current detections; returns the visible tracks. */
  step(dets: Det[]): Track[] {
    for (const t of this.tracks) this.predict(t);
    const high = dets.filter((d) => d.score >= this.o.highThresh);
    const low = dets.filter((d) => d.score >= this.o.lowThresh && d.score < this.o.highThresh);

    // Stage 1 — all tracks vs high-confidence detections
    const s1 = match(this.tracks, high, this.o.iouMatch);
    for (const m of s1.matched) this.update(this.tracks[m.t], high[m.d]);

    // Stage 2 — the ByteTrack recovery: leftover tracks vs LOW-confidence dets
    const leftover = s1.tFree.map((i) => this.tracks[i]);
    const s2 = match(leftover, low, this.o.iouMatch);
    for (const m of s2.matched) this.update(leftover[m.t], low[m.d]);

    // unmatched high detections become new tracks
    for (const j of s1.dFree) this.spawn(high[j]);

    // retire tracks unseen for too long
    this.tracks = this.tracks.filter((t) => t.since <= this.o.maxAge);

    return this.tracks.filter((t) => t.confirmed && t.since === 0);
  }

  get count() { return this.tracks.filter((t) => t.confirmed).length; }
  reset() { this.tracks = []; this.nextId = 1; }
}
