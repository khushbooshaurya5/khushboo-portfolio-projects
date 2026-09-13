# Private RAG in Your Browser

A complete retrieval-augmented-generation pipeline that runs **100% client-side** —
no API key, no server, nothing uploaded.

**Pipeline** (`rag.ts` is the dependency-free, unit-tested core):
1. **Chunk** documents into overlapping, sentence-aware passages.
2. **Embed** each chunk with MiniLM (`Xenova/all-MiniLM-L6-v2`) via transformers.js.
3. **Retrieve** the top-k chunks for a query by cosine similarity.
4. **Generate** a grounded answer with a local LLM (`Qwen2.5-0.5B-Instruct`) that
   runs on the GPU via WebGPU (WASM fallback), instructed to answer only from the
   retrieved context and cite sources as `[n]`.

Paste your own text to index it. Models download once to the browser and are cached;
queries and documents never leave the page.

`rag.ts` = the RAG logic; `main.ts` = transformers.js wiring + UI.
