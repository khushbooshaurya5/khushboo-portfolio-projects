import torch, torch.nn as nn, torch.nn.functional as F, json, os, numpy as np
import torchvision, torchvision.transforms as T
from PIL import Image, ImageDraw
torch.manual_seed(0); np.random.seed(0)
OUT="/home/user/build/ml/anomaly/out"; os.makedirs(OUT,exist_ok=True); os.makedirs(OUT+"/gallery",exist_ok=True)
def load(name):
    C={"FashionMNIST":torchvision.datasets.FashionMNIST,"MNIST":torchvision.datasets.MNIST}[name]
    return C('/tmp/claude-0/'+name,train=True,download=True,transform=T.ToTensor()), C('/tmp/claude-0/'+name,train=False,download=True,transform=T.ToTensor())
try:
    tr,te=load("FashionMNIST"); DS="FashionMNIST"
    FCLASSES=["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]; NORMAL=1  # Trouser
except Exception as e:
    print("FashionMNIST failed, using MNIST:",str(e)[:80],flush=True)
    tr,te=load("MNIST"); DS="MNIST"; FCLASSES=[str(i) for i in range(10)]; NORMAL=1
print("dataset",DS,"normal class",FCLASSES[NORMAL],flush=True)
# normal-only training set
Xtr=torch.stack([tr[i][0] for i in range(len(tr)) if tr[i][1]==NORMAL])
print("normal train samples",len(Xtr),flush=True)
dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr),128,shuffle=True)
class AE(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU())  #7x7
        s.dec=nn.Sequential(nn.ConvTranspose2d(32,16,3,2,1,1),nn.ReLU(),nn.ConvTranspose2d(16,1,3,2,1,1),nn.Sigmoid())
    def forward(s,x): return s.dec(s.enc(x))
ae=AE(); opt=torch.optim.Adam(ae.parameters(),1e-3)
for ep in range(20):
    ae.train(); tot=0
    for (x,) in dl:
        opt.zero_grad(); r=ae(x); loss=F.mse_loss(r,x); loss.backward(); opt.step(); tot+=loss.item()*len(x)
    if (ep+1)%5==0: print(f"epoch {ep+1}/20 mse={tot/len(Xtr):.4f}",flush=True)
ae.eval()
# threshold from normal test recon errors
def err(x):
    with torch.no_grad(): r=ae(x.unsqueeze(0)); return F.mse_loss(r,x.unsqueeze(0)).item()
normal_test=[te[i][0] for i in range(len(te)) if te[i][1]==NORMAL][:400]
errs=np.array([err(x) for x in normal_test]); thr=float(errs.mean()+2*errs.std())
print("normal err mean",errs.mean(),"thr",thr,flush=True)
# export ONNX
torch.onnx.export(ae,torch.randn(1,1,28,28),OUT+"/model.onnx",input_names=["input"],output_names=["recon"],
    dynamic_axes={"input":{0:"b"},"recon":{0:"b"}},opset_version=13)
# gallery: normals, other-class anomalies, synthetic-defect normals
def save(img,name): Image.fromarray((img.squeeze().numpy()*255).astype('uint8')).save(OUT+"/gallery/"+name)
gal=[]; k=0
# normals
for i in range(len(te)):
    if te[i][1]==NORMAL: save(te[i][0],f"{k}.png"); gal.append({"file":f"{k}.png","kind":"normal","label":FCLASSES[NORMAL]}); k+=1
    if k>=4: break
# other-class anomalies
seen=set()
for i in range(len(te)):
    c=te[i][1]
    if c!=NORMAL and c not in seen:
        save(te[i][0],f"{k}.png"); gal.append({"file":f"{k}.png","kind":"anomaly","label":FCLASSES[c]}); seen.add(c); k+=1
    if len(seen)>=4: break
# synthetic-defect normals (add scratch/blob to a trouser)
cnt=0
for i in range(len(te)):
    if te[i][1]==NORMAL:
        arr=(te[i][0].squeeze().numpy()*255).astype('uint8'); im=Image.fromarray(arr); d=ImageDraw.Draw(im)
        if cnt%2==0: d.line([(np.random.randint(6,22),np.random.randint(4,10)),(np.random.randint(6,22),np.random.randint(18,26))],fill=255,width=2)
        else: d.rectangle([np.random.randint(8,16),np.random.randint(8,16),np.random.randint(17,22),np.random.randint(17,22)],fill=0)
        im.save(OUT+f"/gallery/{k}.png"); gal.append({"file":f"{k}.png","kind":"defect","label":FCLASSES[NORMAL]+" + defect"}); k+=1; cnt+=1
    if cnt>=4: break
json.dump({"dataset":DS,"normal":FCLASSES[NORMAL],"threshold":thr,"gallery":gal},open(OUT+"/meta.json","w"))
print("DONE items",k,"onnx",os.path.getsize(OUT+"/model.onnx"),flush=True)
