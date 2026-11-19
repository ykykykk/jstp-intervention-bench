"""Illustrate a trace drawn from frozen commitments. Selection is deliberately illustrative, not a metric."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from lategoal.rig import Rig,EXAM,goal_wins,lane_travel
from lategoal.control import fingerprint
from mechanism.data import read
D=Path(__file__).resolve().parent;E=D.parent/"evidence"

def make():
    dest=D/"figures";dest.mkdir(exist_ok=True)
    p=E/"fresh";s=read(p/"sealed/partial-6043.npz");gold=read(p/"observed/camera.npz");ids=read(p/"sealed/items.npz")["ids"]
    u=s["u"];good=np.where(gold["left"]@EXAM.T>0,-1,1);donor=good[:,[1,0,3,2]]
    condition=(good!=donor)&(u[...,0]==good)&(u[...,1]==donor)&(u[...,2]==good)
    i,state,g=map(int,np.argwhere(condition)[0]);r=Rig(int(ids[i]));w,gear=divmod(state,2);probe=int(s["probe"][i]);side=lambda v:"A" if v<0 else "B"
    images=[];observed=[]
    for j in (0,1):
        a=r.world(w);peak=a.probe(probe);a.set_clutch(gear);cmd=int(u[i,state,g,j]);end=a.stroke(cmd)
        if fingerprint(end)!=gold["frame_sha"][i,state,int(cmd>0)]:raise ValueError("插图与真实封卷画面不同")
        images.append(end);observed.append(bool(goal_wins(lane_travel(r.base,end),EXAM[g])))
    record=dict(layout=int(ids[i]),state=state,goal=EXAM[g].tolist(),probe=probe,native=int(u[i,state,g,0]),
            physical=int(u[i,state,g,1]),null=int(u[i,state,g,2]),actual_success=observed)
    (dest/"case.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf8",newline="\n")
    pic=Image.new("RGB",(1056,264),"#f2f6f2");dr=ImageDraw.Draw(pic)
    fontfile=D/"fonts/JSTPSerif-Regular.ttf";font=ImageFont.truetype(str(fontfile),21);small=ImageFont.truetype(str(fontfile),16)
    seen=[peak,r.base,*images];labels=["横向试探 "+side(probe),"同一复位画面","原预期：拉 "+side(record["native"]),"改预期：拉 "+side(record["physical"])]
    for j,(im,label) in enumerate(zip(seen,labels)):
        l=12+262*j;dr.text((l+2,10),label,fill="#143333",font=font);pic.paste(Image.fromarray(im).resize((248,166),Image.Resampling.NEAREST),(l,51))
    dr.text((14,226),"真实相机",fill="#365c56",font=small);dr.text((536,226),"实际成功",fill="#116655",font=small);dr.text((798,226),"原世界失败",fill="#a04432",font=small)
    pic.save(dest/"frozen-case.png")

if __name__=="__main__":make()

