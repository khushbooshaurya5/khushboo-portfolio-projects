import torch, torch.nn as nn, torch.nn.functional as F, json, os
import torchvision, torchvision.transforms as T
from PIL import Image
torch.manual_seed(1); torch.set_num_threads(4)
OUT="/home/user/build/ml/gradcam/out"; os.makedirs(OUT,exist_ok=True); os.makedirs(OUT+"/gallery",exist_ok=True)
CLASSES=["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
tr=torchvision.datasets.FashionMNIST('/tmp/claude-0/FashionMNIST',train=True,download=True,transform=T.ToTensor())
te=torchvision.datasets.FashionMNIST('/tmp/claude-0/FashionMNIST',train=False,download=True,transform=T.ToTensor())
sub=torch.utils.data.Subset(tr, torch.randperm(len(tr))[:16000].tolist())
tl=torch.utils.data.DataLoader(sub,128,shuffle=True,num_workers=0); vl=torch.utils.data.DataLoader(te,2000,num_workers=0)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return [nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
        s.features=nn.Sequential(*blk(1,16),nn.MaxPool2d(2),*blk(16,32),nn.MaxPool2d(2),*blk(32,32)) # 7x7x32
        s.fc=nn.Linear(32,10)
    def forward(s,x):
        f=s.features(x); p=F.adaptive_avg_pool2d(f,1).flatten(1); return s.fc(p), f
net=Net(); opt=torch.optim.Adam(net.parameters(),1e-3)
for ep in range(5):
    net.train()
    for x,y in tl:
        opt.zero_grad(); o,_=net(x); F.cross_entropy(o,y).backward(); opt.step()
    net.eval(); c=t=0
    with torch.no_grad():
        for x,y in vl:
            o,_=net(x); c+=(o.argmax(1)==y).sum().item(); t+=len(y)
    print(f"epoch {ep+1}/5 acc={c/t:.3f}",flush=True)
acc=c/t; net.eval()
torch.onnx.export(net,torch.randn(1,1,28,28),OUT+"/model.onnx",input_names=["input"],output_names=["logits","featmap"],dynamic_axes={"input":{0:"b"},"logits":{0:"b"},"featmap":{0:"b"}},opset_version=17,dynamo=False)
import onnx
m=onnx.load(OUT+"/model.onnx"); onnx.save_model(m,OUT+"/model.onnx",save_as_external_data=False)
for f in os.listdir(OUT):
    if f.endswith(".data"): os.remove(OUT+"/"+f)
json.dump({"fc_weight":net.fc.weight.detach().tolist(),"classes":CLASSES,"acc":round(acc,4)},open(OUT+"/cam.json","w"))
per={}; gal=[]; k=0
for i in range(len(te)):
    x,y=te[i]
    if per.get(y,0)<2:
        per[y]=per.get(y,0)+1
        Image.fromarray((x.squeeze().numpy()*255).astype('uint8')).save(OUT+f"/gallery/{k}.png")
        gal.append({"file":f"{k}.png","label":CLASSES[y]}); k+=1
    if k>=20: break
json.dump(gal,open(OUT+"/gallery.json","w"))
print("DONE acc",acc,"onnx",os.path.getsize(OUT+"/model.onnx"),flush=True)
