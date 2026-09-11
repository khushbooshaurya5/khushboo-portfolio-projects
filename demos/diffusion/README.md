# 2D Diffusion — watch it denoise

A tiny **class-conditional DDPM** (PyTorch) trained to turn Gaussian noise into
2D target distributions. One MLP (`train.py`) predicts the noise added at each of
200 timesteps; the browser runs the **reverse process** on 1,600 particles via
onnxruntime-web, animating noise → shape. Four shapes from a single model:
spiral, heart, star, flower.

- Model + inference run **entirely client-side** (ONNX, shared wasm in `../_ort/`).
- `train.py` is the full training + ONNX export script.
