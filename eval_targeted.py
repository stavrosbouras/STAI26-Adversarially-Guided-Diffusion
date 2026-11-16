"""Pixel reassignment in a targeted run.

Takes the pixels that the segmentation network predicts as the source class on the clean
images and counts how many are predicted as the target class, as another class, or still
as the source class on the adversarial images. Only pixels valid in both images are
counted. Output goes to targeted_results.txt in the run directory, or to -o.

Example:
    python eval_targeted.py --run_dir targeted_road_to_sidewalk --source 9 --target 11 --seg_net cenet
"""
import os, sys, math, numpy as np, torch, torch.nn as nn, argparse
sys.path.insert(0,'.')
from lidm.eval import build_model
from segnets.cenet import ResNet_34 as CENet

FOV = [3.0, -25.0]
DEPTH_SCALE = 56.0
RANGENET_HEAD = 'pretrained_weights/kitti/rangenet/segmentation_head'
CENET_WEIGHTS = 'pretrained_weights/kitti/cenet/SalsaNext_valid_best'
RN_MEAN = np.array([12.12,10.88,0.23,-1.04],dtype=np.float32)
RN_STD  = np.array([12.32,11.47,6.91,0.86],dtype=np.float32)
CN_MEAN = np.array([11.71279,-0.1023471,0.4952,-1.0545,0.2877],dtype=np.float32)
CN_STD  = np.array([10.24,12.295865,9.4287,0.8643,0.1450],dtype=np.float32)


def range2xyz(r01):
    if r01.dim()==4: r01=r01.squeeze(1)
    B,H,W=r01.shape; dev=r01.device
    fu=FOV[0]/180*math.pi; fd=FOV[1]/180*math.pi; fr=abs(fd)+abs(fu)
    depth=r01*DEPTH_SCALE
    sx=torch.linspace(0,1,W,device=dev).view(1,1,W).expand(B,H,-1)
    sy=torch.linspace(0,1,H,device=dev).view(1,H,1).expand(B,-1,W)
    yaw=math.pi*(sx*2-1); pitch=(1.0-sy)*fr-abs(fd)
    return torch.stack([depth, torch.cos(yaw)*torch.cos(pitch)*depth,
                        -torch.sin(yaw)*torch.cos(pitch)*depth,
                        torch.sin(pitch)*depth], dim=1)


class RNwH(nn.Module):
    def __init__(self,b,h):
        super().__init__(); self.backbone=b.backbone; self.decoder=b.decoder
        self.head=nn.Conv2d(32,20,3,padding=1)
        d=torch.load(h,map_location='cpu'); self.head.weight.data=d['1.weight']; self.head.bias.data=d['1.bias']
    def forward(self,x):
        y,s=self.backbone(x); y=self.decoder(y,s); return self.head(y)


def build_segnet(seg_net, device):
    if seg_net=='rangenet':
        base=build_model('kitti','rangenet',device='cpu')
        return RNwH(base,RANGENET_HEAD).to(device).eval(),'rangenet'
    m=CENet(nclasses=20, params={'train':{'aux_loss':{'use':True}}}).to(device)
    ck=torch.load(CENET_WEIGHTS,map_location='cpu',weights_only=False)
    sd=ck.get('state_dict',ck) if isinstance(ck,dict) else ck
    sd={k.replace('module.','',1):v for k,v in sd.items()}
    m.load_state_dict(sd,strict=False); m.eval()
    return m,'cenet'


def predict(model, net_type, img, device):
    r01=(img.squeeze(1)+1.0)/2.0
    dxz=range2xyz(r01).to(device)
    v=(dxz[:,0:1]>1.0).float()
    if net_type=='rangenet':
        m=torch.from_numpy(RN_MEAN).to(device).view(1,4,1,1)
        s=torch.from_numpy(RN_STD).to(device).view(1,4,1,1)
        x=(dxz*v+m*(1-v)-m)/s
    else:
        B,_,H,W=dxz.shape
        m=torch.from_numpy(CN_MEAN).to(device).view(1,5,1,1)
        s=torch.from_numpy(CN_STD).to(device).view(1,5,1,1)
        rem=torch.full((B,1,H,W),float(CN_MEAN[4]),device=device)
        x=torch.cat([dxz,rem],dim=1); x=(x*v+m*(1-v)-m)/s
    with torch.no_grad():
        out=model(x)
        if isinstance(out,(list,tuple)): out=out[0]
        pred=out.argmax(1)[0].cpu().numpy()
    valid=(v[0,0].cpu().numpy()>0.5)
    return pred, valid


ap=argparse.ArgumentParser()
ap.add_argument('--run_dir', required=True)
ap.add_argument('--source', type=int, default=9)
ap.add_argument('--target', type=int, default=11)
ap.add_argument('--seg_net', default='cenet')
ap.add_argument('-o', '--output', default=None,
                help='text file for the printed results (default: <run_dir>/targeted_results.txt)')
args=ap.parse_args()

device=torch.device('cuda')
clean=torch.load(f'{args.run_dir}/clean_ranges.pt', weights_only=False)
adv  =torch.load(f'{args.run_dir}/adv_ranges.pt',   weights_only=False)
model,net_type=build_segnet(args.seg_net, device)

tot_src=0; src_to_tgt=0; src_to_other=0; src_stay=0
for i in range(len(clean)):
    cp,cv=predict(model,net_type,clean[i:i+1].to(device),device)
    ap_,av=predict(model,net_type,adv[i:i+1].to(device),device)
    m=(cp==args.source)&cv&av
    tot_src += int(m.sum())
    src_to_tgt   += int(((ap_==args.target)&m).sum())
    src_stay     += int(((ap_==args.source)&m).sum())
    src_to_other += int(((ap_!=args.target)&(ap_!=args.source)&m).sum())

lines=[]
def log(s=''):
    print(s)
    lines.append(s)

log(f'Source=class {args.source}, Target=class {args.target}')
log(f'Total source pixels (clean): {tot_src}')
if tot_src>0:
    log(f'  stayed source:    {src_stay:8d}  ({100*src_stay/tot_src:5.1f}%)')
    log(f'  -> TARGET (goal): {src_to_tgt:8d}  ({100*src_to_tgt/tot_src:5.1f}%)')
    log(f'  -> other classes: {src_to_other:8d}  ({100*src_to_other/tot_src:5.1f}%)')

out_path=args.output or os.path.join(args.run_dir, 'targeted_results.txt')
with open(out_path,'w') as f:
    f.write('\n'.join(lines)+'\n')
print(f'\n[done] saved {out_path}')
