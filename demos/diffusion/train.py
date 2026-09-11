"""
2D denoising-diffusion (DDPM) over several target shapes, class-conditional.
Trains a tiny MLP to predict the noise added to 2D points; the browser runs the
reverse process, animating a cloud of noise collapsing into the chosen shape.

Fast on CPU (tiny model, 2D data). Exports model.onnx + meta.json (schedule,
shapes, target point clouds for a faint reference overlay).
"""
import json, math, numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

torch.manual_seed(0); np.random.seed(0)
OUT = "web/public"
import os; os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# target shapes -> point clouds, normalized to ~unit scale (mean 0, std ~0.6)
# ----------------------------------------------------------------------------
def normalize(P):
    P = P - P.mean(0)
    P = P / (P.std() + 1e-8) * 0.6
    return P.astype(np.float32)

def spiral(n=4000):
    t = np.sqrt(np.random.rand(n)) * 3.2 * math.pi
    r = t / (3.2 * math.pi)
    x = r * np.cos(t); y = r * np.sin(t)
    xy = np.stack([x, y], 1) + np.random.randn(n, 2) * 0.02
    return normalize(xy)

def heart(n=4000):
    t = np.random.rand(n) * 2 * math.pi
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    xy = np.stack([x, y], 1) + np.random.randn(n, 2) * 0.3
    return normalize(xy)

def from_text(txt, n=4000, size=200):
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    fnt = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            fnt = ImageFont.truetype(path, int(size * 0.62)); break
    if fnt is None: fnt = ImageFont.load_default()
    bb = dr.textbbox((0, 0), txt, font=fnt, stroke_width=6)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    dr.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), txt, fill=255,
            font=fnt, stroke_width=6, stroke_fill=255)   # thicker strokes survive diffusion
    a = np.array(img)
    ys, xs = np.where(a > 100)
    idx = np.random.randint(0, len(xs), n)
    xy = np.stack([xs[idx], -ys[idx]], 1).astype(np.float32) + np.random.randn(n, 2) * 0.8
    return normalize(xy)

def star(n=4000, pk=5):
    # 5-point star OUTLINE: walk the star's edges (crisp, like the heart outline)
    R, r = 1.0, 0.40
    verts = []
    for k in range(pk * 2):
        ang = math.pi / 2 + k * math.pi / pk
        rad = R if k % 2 == 0 else r
        verts.append((rad * math.cos(ang), rad * math.sin(ang)))
    pts = []
    for i in range(len(verts)):
        a = np.array(verts[i]); b = np.array(verts[(i + 1) % len(verts)])
        for _ in range(n // (pk * 2)):
            pts.append(a + (b - a) * np.random.rand())
    xy = np.array(pts, np.float32) + np.random.randn(len(pts), 2) * 0.012
    return normalize(xy)

def flower(n=4000, k=3):
    # rose curve r = cos(k*theta): a clean multi-petal outline
    t = np.random.rand(n) * 2 * math.pi
    r = np.cos(k * t)
    xy = np.stack([r * np.cos(t), r * np.sin(t)], 1) + np.random.randn(n, 2) * 0.02
    return normalize(xy)

SHAPES = ["spiral", "heart", "star", "flower"]
CLOUDS = [spiral(6000), heart(6000), star(6000), flower(6000)]
NC = len(SHAPES)

# ----------------------------------------------------------------------------
# DDPM schedule
# ----------------------------------------------------------------------------
T = 200
betas = np.linspace(1e-4, 0.02, T).astype(np.float32)
alphas = 1.0 - betas
abar = np.cumprod(alphas).astype(np.float32)

# ----------------------------------------------------------------------------
# model: eps_theta(x[B,2], t[B,1] in [0,1], c[B,NC]) -> eps[B,2]
# sinusoidal time embedding is computed inside forward() so it lives in the ONNX
# ----------------------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self, temb=48, hid=384):
        super().__init__()
        self.temb = temb
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), temb // 2))
        self.register_buffer("freqs", freqs)
        din = 2 + temb + NC
        self.net = nn.Sequential(
            nn.Linear(din, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, 2),
        )
    def forward(self, x, t, c):
        ang = t * self.freqs.unsqueeze(0)          # [B, temb/2]
        te = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        h = torch.cat([x, te, c], dim=1)
        return self.net(h)

dev = "cpu"
model = Denoiser().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
abar_t = torch.tensor(abar)

clouds_t = [torch.tensor(c) for c in CLOUDS]
B = 1024
STEPS = 14000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
for step in range(STEPS):
    ci = np.random.randint(0, NC)
    pts = clouds_t[ci]
    idx = torch.randint(0, pts.shape[0], (B,))
    x0 = pts[idx]
    t = torch.randint(0, T, (B,))
    ab = abar_t[t].unsqueeze(1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    tn = (t.float() / (T - 1)).unsqueeze(1)
    c = torch.zeros(B, NC); c[:, ci] = 1.0
    pred = model(xt, tn, c)
    loss = ((pred - noise) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0:
        print(f"step {step:5d}/{STEPS}  loss {loss.item():.4f}")

model.eval()

# quick reverse-sampling sanity check (report mean distance of samples to target)
@torch.no_grad()
def sample(ci, n=800):
    x = torch.randn(n, 2)
    c = torch.zeros(n, NC); c[:, ci] = 1.0
    for ti in range(T - 1, -1, -1):
        tn = torch.full((n, 1), ti / (T - 1))
        eps = model(x, tn, c)
        a = alphas[ti]; ab = abar[ti]; b = betas[ti]
        mean = (x - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
        x = mean + (math.sqrt(b) * torch.randn_like(x) if ti > 0 else 0)
    return x.numpy()

for ci, nm in enumerate(SHAPES):
    s = sample(ci)
    print(f"  {nm}: sample std {s.std():.3f} (target ~0.6)")

# ----------------------------------------------------------------------------
# export ONNX
# ----------------------------------------------------------------------------
dummy = (torch.randn(4, 2), torch.rand(4, 1), torch.eye(NC)[:4])
torch.onnx.export(
    model, dummy, f"{OUT}/model.onnx",
    input_names=["x", "t", "c"], output_names=["eps"],
    dynamic_axes={"x": {0: "n"}, "t": {0: "n"}, "c": {0: "n"}, "eps": {0: "n"}},
    opset_version=17, dynamo=False,
)

# consolidate any external data into the single .onnx
import onnx
m = onnx.load(f"{OUT}/model.onnx")
onnx.save_model(m, f"{OUT}/model.onnx", save_as_external_data=False)
for f in os.listdir(OUT):
    if f.endswith(".onnx.data") or f == "model.onnx_data":
        os.remove(os.path.join(OUT, f))

# meta: schedule + shapes + a faint reference cloud (downsampled) per shape
def thin(c, k=600):
    i = np.random.choice(len(c), k, replace=False)
    return np.round(c[i], 3).tolist()

meta = {
    "T": T,
    "betas": betas.round(6).tolist(),
    "alphas": alphas.round(6).tolist(),
    "abar": abar.round(6).tolist(),
    "shapes": SHAPES,
    "targets": [thin(c) for c in CLOUDS],
}
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print("wrote model.onnx + meta.json")
