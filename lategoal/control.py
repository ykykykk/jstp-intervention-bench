"""动作选择不进口物理裁判；区域方向、成功标签、供体动作都不送入控制器。"""
import hashlib,json
import numpy as np
import torch
from .rig import C,REL,PROTO,Rig,first,history,read,sha
from .net import Net,sign
MODES=("native","repeat","physical","null","equivalent","full")
def T(x):return torch.as_tensor(x,dtype=torch.float32)
def fingerprint(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest().encode()
def load_net(rel,arm,seed):
    p=rel/"models"/f"{arm}-{seed}";meta=json.loads((p/"freeze.json").read_text(encoding="utf8"))
    if meta["weights"]!=sha(p/"last.pt") or meta["protocol"]!=sha(PROTO):raise ValueError("冻结指纹不符")
    net=Net().eval();net.load_state_dict(torch.load(p/"last.pt",map_location="cpu",weights_only=True));return net
@torch.inference_mode()
def collect(net,rigs):
    probe=sign(net.probe(T(np.stack([first(r) for r in rigs])))).numpy().astype("int8")
    xs=[];peak_sha=[];base_sha=[]
    for r,p in zip(rigs,probe):
        peaks=[r.world(s).probe(p) for s in (0,1)]
        xs.append([history(r,peak,p,g) for peak in peaks for g in (0,1)])
        peak_sha.append([fingerprint(peak) for peak in peaks]);base_sha.append(fingerprint(r.base))
    hist=T(np.stack(xs));n=len(hist)
    h=net.observe(hist.flatten(0,1));z=net.future(h,T(np.repeat(probe,4)))
    return dict(h=h.reshape(n,4,-1),z=z.reshape(n,4,-1),p=probe,peak=np.asarray(peak_sha),base=np.asarray(base_sha))
def axes(measured,net):
    m=measured.reshape(-1,12).astype('float64');_,ev,vt=np.linalg.svd(m-m.mean(0),full_matrices=False)
    pc=vt[:C["rank"]];W=net.eye.weight.detach().numpy().astype('float64')
    _,sv,vt=np.linalg.svd(pc@W,full_matrices=False)
    if sv[-1]<1e-8:raise ValueError("物理读出不足四维")
    Q=vt.T
    _,ss,vt=np.linalg.svd(W,full_matrices=True);kernel=vt[(ss>1e-9).sum():].T
    rn=np.random.default_rng(730420).normal(size=(kernel.shape[1],Q.shape[1]))
    N=kernel@np.linalg.qr(rn)[0]
    return dict(Q=Q.astype('float32'),N=N.astype('float32'),pc=pc.astype('float32'),
                variance=np.array((ev[:len(pc)]**2).sum()/(ev**2).sum()))
@torch.inference_mode()
def decisions(net,arm,c,goals,basis):
    h,z=c['h'],c['z'];n=len(h);G=len(goals);gg=T(goals)
    donor=z[:,[1,0,3,2]];equiv=z[:,[3,2,1,0]]
    if basis is None:
        physical=null=z
    else:
        q,nul=T(basis["Q"]),T(basis["N"]);coeff=(donor-z)@q
        physical=z+coeff@q.T;null=z+coeff@nul.T
    variants=(z,z.clone(),physical,null,equiv,donor);us=[];sights=[]
    for v in variants:
        # h 原封不动；一次编码分给所有目标。没有用 eye() 的数值来选动作。
        hv=h[:,:,None,:].expand(-1,-1,G,-1).reshape(n*4*G,-1)
        zv=v[:,:,None,:].expand(-1,-1,G,-1).reshape(n*4*G,-1)
        goal=gg[None,None].expand(n,4,-1,-1).reshape(n*4*G,-1)
        u=sign(net.issue(hv,zv,goal,arm)).reshape(n,4,G);us.append(u.numpy().astype('int8'))
        sights.append(net.eye(v).numpy())
    norms=torch.stack(((physical-z).norm(dim=-1),(null-z).norm(dim=-1)),dim=-1).numpy()
    return dict(u=np.stack(us,-1),eye=np.stack(sights,2),norm=norms,probe=c["p"],peak=c["peak"],base=c["base"],
        prefix=np.asarray([[fingerprint(torch.cat((hi,zi)).numpy()) for hi,zi in zip(hr,zr)] for hr,zr in zip(h,z)]))
