# Prune Latency Bench (WASM vs WebGPU)

Companion tool to the paper *"Sparsity Is Not Speed: Measuring the Real
In-Browser Latency of Pruned Convolutional Networks."*

Open `index.html` (served over HTTP, e.g. GitHub Pages) and press **Run**: it
times real inference of five FashionMNIST CNN variants — dense, structured 50/70/90%,
and unstructured 90% — on **both** the WASM and WebGPU backends of the visitor's
own device, via onnxruntime-web. Results (with device info) can be copied as JSON
to extend the paper's device matrix.

Models in `models/` are the exact ONNX exports measured in the paper.
