/**
 * Client-side RAG core (dependency-free, framework-agnostic).
 * Chunking, cosine retrieval, and grounded-prompt assembly — the parts that make
 * a RAG system correct. The embedding and generation models are supplied by the
 * caller (transformers.js), so this module is pure and unit-testable.
 */

export interface Chunk { id: number; text: string; source: string; vec?: Float32Array; }
export interface Hit { chunk: Chunk; score: number; }

/** Split a document into overlapping, sentence-aware chunks (~`size` words, `overlap` words shared). */
export function chunk(text: string, source: string, size = 60, overlap = 15, startId = 0): Chunk[] {
  const sentences = text.replace(/\s+/g, " ").trim().match(/[^.!?]+[.!?]*/g) || [text];
  const chunks: Chunk[] = [];
  let buf: string[] = [], words = 0, id = startId;
  const flush = () => {
    if (!buf.length) return;
    chunks.push({ id: id++, text: buf.join(" ").trim(), source });
  };
  for (const s of sentences) {
    const w = s.trim().split(/\s+/).length;
    buf.push(s.trim()); words += w;
    if (words >= size) {
      flush();
      // start next chunk with the tail of this one for context overlap
      const tail: string[] = []; let tw = 0;
      for (let i = buf.length - 1; i >= 0 && tw < overlap; i--) { tail.unshift(buf[i]); tw += buf[i].split(/\s+/).length; }
      buf = tail.slice(); words = tw;
    }
  }
  flush();
  return chunks;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

/** Rank chunks by cosine similarity to the query vector; return the top-k. */
export function topK(query: Float32Array, chunks: Chunk[], k = 4): Hit[] {
  return chunks
    .filter((c) => c.vec)
    .map((c) => ({ chunk: c, score: cosine(query, c.vec!) }))
    .sort((x, y) => y.score - x.score)
    .slice(0, k);
}

/** Assemble a grounded chat prompt: the model must answer only from the numbered context and cite [n]. */
export function buildMessages(question: string, hits: Hit[]) {
  const context = hits.map((h, i) => `[${i + 1}] (${h.chunk.source}) ${h.chunk.text}`).join("\n\n");
  const system =
    "You are a retrieval-augmented assistant. Answer the question using ONLY the numbered context below. " +
    "Cite the sources you use inline as [1], [2], etc. If the context does not contain the answer, say you don't know. " +
    "Be concise.";
  const user = `Context:\n${context}\n\nQuestion: ${question}`;
  return [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
}
