"""独立重放前期证据。V6 为归档推理复现，不宣称从头重训旧项目。"""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,read_json,sha,new_json

@torch.inference_mode()
def replay6(device):
    if not device.startswith("cuda"):raise ValueError("V6 原管线用 CUDA FP16/BF16；CPU 不充作精确复现。")
    from .legacy6.grounded_state import GroundedConfig,GroundedPredictor
    from .legacy6.world import GroundedEnvironment
    from .legacy6.planning import forecast
    d=ROOT/"evidence/legacy6";ck=torch.load(d/"inference.pt",map_location="cpu",weights_only=True)
    if sha(d/"inference.pt")!=read_json(d/"source.json")["stripped_checkpoint"]:raise ValueError("V6 推理权重被改变")
    model=GroundedPredictor(GroundedConfig.from_dict(ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"]); refs=read_json(d/"trials.json"); mismatches=[]; totals={}
    for row in refs:
        env=GroundedEnvironment(row["seed"],row["level"]);prior=state=prev=None
        for step in range(1,81):
            im=torch.from_numpy(env.camera()[None]).permute(0,3,1,2).float().div_(255).to(device)
            body=torch.from_numpy(env.world.proprio()[None].copy()).to(device)
            with torch.autocast("cuda",dtype=torch.float16):
                if prev is not None:prior=model.transition(state,torch.tensor([prev],device=device),torch.tensor([.2],device=device))
                state=model.observe(im,body,prior);logits=model.action_head(state)
            if row["role"]=="direct":act=int(logits.float().softmax(-1)[0].argmax())
            else:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    out=forecast(model,state,horizon=ck["selected_horizon"],wrong=row["role"]=="wrong_future",prior_weight=ck["selected_prior_weight"])
                k=int(out["scores"].argmax());act=int(out["actions"][k,0])
            env.world.apply(act);prev=act
            if env.world.success():break
        won=bool(env.world.success());totals[row["role"]]=totals.get(row["role"],0)+int(won)
        if [won,step]!=[row["success"],row["steps"]]:mismatches.append([row["index"],row["role"],won,step])
    return dict(n=len(refs),totals=totals,mismatched_trials=mismatches)

@torch.inference_mode()
def replay7(device):
    from .v7_world import setup,World,TABLE
    from .v7_model import JointFuture,rgb,goal_cost,picture,displacement
    d=ROOT/"evidence/legacy7"; meta=read_json(d/"freeze.json")
    if sha(d/"frozen.pt")!=meta["sha256"]:raise ValueError("V7 原始冻结权重指纹改变")
    m=JointFuture().to(device).eval();m.load_state_dict(torch.load(d/"frozen.pt",map_location=device,weights_only=True)["weights"])
    refs=[json.loads(l) for l in (d/"choices_before_action.jsonl").read_text(encoding="utf8").splitlines()]
    successes=[0,0,0]; hits=0; switches=0;mismatches=0;floats=0.
    for r in refs:
        b,p,a,dy=setup(r["seed"]);goal=b.goals(dy); histories=[World(b,j).probe(p) for j in (0,1)];cur=histories[0][2]
        c=rgb(cur,device)[None].expand(2,-1,-1,-1); prop=torch.as_tensor(p,device=device).float()[None].expand(2,-1)
        opts=torch.as_tensor(a,device=device).float(); read=[]; obs=[]
        for j,h in enumerate(histories):
            ys=[];preds=[]
            for g in (0,1):
                gt=rgb(goal[g],device)[None,None]; ks=[]
                for mode,peak in enumerate((h[1],cur,histories[1-j][1])):
                    ep=rgb(peak,device)[None].expand(2,-1,-1,-1); f=m(c,ep,c,prop,opts)
                    k=int(goal_cost(f[None],gt,c[:1])[0,0].argmin());ks.append(k)
                    w=World(b,j);w.probe(p);actual=w.execute(a[k]);successes[mode]+=int(np.array_equal(actual[:TABLE],goal[g]))
                    if mode==0:preds.append(f)
                for mode,key in enumerate(("selected","empty_evidence_selected","exchanged_evidence_selected")):
                    mismatches+=int(ks[mode]!=r[key][j][g])
                ys.append(ks[0])
            switches+=int(ys[0]!=ys[1]); floats=max(floats,float((preds[0]-preds[1]).abs().max()))
            read.append(displacement(picture(preds[0]),cur)[...,1,1]*np.sign(dy))
            actual=[]
            for opt in a:
                w=World(b,j);w.probe(p);actual.append(w.execute(opt))
            obs.append(displacement(np.stack(actual),cur)[...,1,1]*np.sign(dy))
        hits+=int(((np.asarray(read)>=3)==(np.asarray(obs)>=3)).all(0).sum())
    return dict(layouts=len(refs),factual_flat_exchange=successes,physics_pairs=hits,switches=switches,goal_delta=floats,choice_mismatches=mismatches)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("version",choices=["v6","v7"]);p.add_argument("--device",default="cuda");p.add_argument("--out",type=Path)
    args=p.parse_args();torch.set_num_threads(4);os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
    result=(replay6 if args.version=="v6" else replay7)(args.device);print(json.dumps(result),flush=True)
    if args.out:new_json(args.out,result)
