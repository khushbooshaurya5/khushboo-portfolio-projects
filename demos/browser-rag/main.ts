import { pipeline, env, TextStreamer } from "@huggingface/transformers";
import { chunk, topK, buildMessages, type Chunk, type Hit } from "./rag";

env.allowLocalModels = false;   // always fetch from the HF Hub CDN

const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const GEN_MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $("status");
const deviceEl = $("device");
const hitsEl = $("hits");
const answerEl = $("answer");
const qinput = $("q") as HTMLInputElement;
const askBtn = $("ask") as HTMLButtonElement;
const genBtn = $("gen") as HTMLButtonElement;
const docsInput = $("docs") as HTMLTextAreaElement;
const rebuildBtn = $("rebuild") as HTMLButtonElement;
const kbInfo = $("kbinfo");

const DEVICE: "webgpu" | "wasm" = (navigator as any).gpu ? "webgpu" : "wasm";

// A small, self-demonstrating knowledge base (about the ML this portfolio is built on).
const SEED: { source: string; text: string }[] = [
  { source: "pruning.md", text: "Structured pruning removes whole units such as convolutional channels, so the network genuinely becomes smaller and its dense compute drops — which yields a real inference speedup. Unstructured (magnitude) pruning only sets individual weights to zero; the tensor shapes are unchanged, so on a dense runtime with no sparse kernels the zeroed weights are still multiplied and there is little or no speedup." },
  { source: "quantization.md", text: "Quantization stores and computes weights at lower precision, such as int8 instead of fp32. It shrinks the model roughly four times and can speed up inference on hardware with integer kernels. It trades a small amount of accuracy for large gains in size and, on supported runtimes, latency." },
  { source: "rag.md", text: "Retrieval-augmented generation (RAG) grounds a language model in external documents. The pipeline embeds each chunk of the corpus into a vector, embeds the user query the same way, retrieves the most similar chunks by cosine similarity, and puts them in the prompt so the model answers from real sources and can cite them. This reduces hallucination compared with a bare LLM." },
  { source: "embeddings.md", text: "An embedding maps text to a dense vector so that semantically similar texts land close together. Sentence embedding models like MiniLM are small and fast, which makes them ideal for on-device retrieval. Cosine similarity between the query vector and document vectors ranks relevance." },
  { source: "webgpu.md", text: "WebGPU runs neural-network compute on the GPU from inside the browser, while WebAssembly (WASM) runs it on the CPU. For large models WebGPU is much faster, but for very small models its fixed data-transfer and kernel-launch overhead can make it slower than WASM. Both let models run fully client-side with no server." },
  { source: "onnx.md", text: "ONNX is an open format for exporting trained neural networks so they run across different runtimes. ONNX Runtime Web executes ONNX models in the browser on WASM or WebGPU, which is how a PyTorch model trained offline can run client-side with no backend and no data leaving the device." },
];

let embedder: any = null;
let generator: any = null;
let chunks: Chunk[] = [];
let lastHits: Hit[] = [];

async function embedText(text: string): Promise<Float32Array> {
  const out = await embedder(text, { pooling: "mean", normalize: true });
  return out.data as Float32Array;
}
async function embedChunks(cs: Chunk[]) {
  for (let i = 0; i < cs.length; i++) { cs[i].vec = await embedText(cs[i].text); statusEl.textContent = `embedding knowledge base… ${i + 1}/${cs.length}`; }
}

async function buildIndex() {
  const extra = docsInput.value.trim();
  chunks = [];
  let id = 0;
  for (const d of SEED) { const cs = chunk(d.text, d.source, 55, 12, id); id += cs.length; chunks.push(...cs); }
  if (extra) { const cs = chunk(extra, "your-notes", 55, 12, id); id += cs.length; chunks.push(...cs); }
  statusEl.textContent = "embedding knowledge base…";
  await embedChunks(chunks);
  kbInfo.textContent = `${chunks.length} chunks indexed · ${SEED.length}${extra ? " + your" : ""} sources`;
  statusEl.textContent = "ready — ask a question.";
  askBtn.disabled = false;
}

function renderHits(hits: Hit[]) {
  hitsEl.innerHTML = hits.map((h, i) =>
    `<div class="hit"><div class="hh"><span class="hn">[${i + 1}]</span><span class="hsrc">${h.chunk.source}</span><span class="hsc">${h.score.toFixed(3)}</span></div><div class="ht">${h.chunk.text}</div></div>`
  ).join("");
}

async function retrieve() {
  const q = qinput.value.trim();
  if (!q || !embedder) return;
  statusEl.textContent = "retrieving…";
  const qv = await embedText(q);
  lastHits = topK(qv, chunks, 4);
  renderHits(lastHits);
  genBtn.disabled = false;
  answerEl.innerHTML = "";
  statusEl.textContent = `retrieved ${lastHits.length} chunks — click “Generate grounded answer”.`;
}

async function generate() {
  if (!lastHits.length) return;
  genBtn.disabled = true;
  answerEl.textContent = "";
  try {
    if (!generator) {
      statusEl.textContent = `loading local LLM (${GEN_MODEL.split("/")[1]}) — one-time download…`;
      generator = await pipeline("text-generation", GEN_MODEL, {
        device: DEVICE, dtype: DEVICE === "webgpu" ? "q4f16" : "q4",
        progress_callback: (p: any) => { if (p.status === "progress" && p.total) statusEl.textContent = `downloading LLM… ${(p.progress || 0).toFixed(0)}%`; },
      });
    }
    statusEl.textContent = "generating grounded answer…";
    const messages = buildMessages(qinput.value.trim(), lastHits);
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true, skip_special_tokens: true,
      callback_function: (t: string) => { answerEl.textContent += t; },
    });
    await generator(messages, { max_new_tokens: 320, do_sample: false, streamer });
    statusEl.textContent = "done — answer grounded in the retrieved chunks.";
  } catch (e: any) {
    statusEl.textContent = "⚠ generation failed (" + (e?.message || e) + "). Retrieval above still works.";
  } finally {
    genBtn.disabled = false;
  }
}

async function main() {
  deviceEl.textContent = DEVICE === "webgpu" ? "WebGPU" : "WASM (CPU)";
  statusEl.textContent = "loading embedding model…";
  try {
    embedder = await pipeline("feature-extraction", EMBED_MODEL, { device: DEVICE });
  } catch {
    embedder = await pipeline("feature-extraction", EMBED_MODEL);   // fall back to default backend
  }
  await buildIndex();
  askBtn.addEventListener("click", retrieve);
  qinput.addEventListener("keydown", (e) => { if (e.key === "Enter") retrieve(); });
  genBtn.addEventListener("click", generate);
  rebuildBtn.addEventListener("click", () => { askBtn.disabled = true; genBtn.disabled = true; buildIndex(); });
}
main().catch((e) => { statusEl.textContent = "⚠ " + (e?.message || e); console.error(e); });
