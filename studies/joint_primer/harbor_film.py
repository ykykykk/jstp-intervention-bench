"""Make three synchronized game films from harbor_capture's camera/decision records.

The normal Switchboard draws its boats, berths, controls and splashes. This raster
target exports those drawing commands, not a screenshot of the user's Windows desktop.
"""
import argparse
from functools import lru_cache
import json
from math import hypot
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .game_board import Switchboard, HANDLE, SEA, GOLD, BLUE, WHITE, MUTED, RED, smooth
from .game_round import horizontal_centres
from .harbor_capture import ARMS, BUDGETS, NAMES
from lategoal.rig import Rig


W,H,FPS,SECONDS=1600,900,20,32
BG,BOX,EDGE="#0a1924","#152a38","#30515d"
FONT_FOLDER=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts"


@lru_cache(maxsize=32)
def font(size,bold=False,latin=False):
    file=(("segoeuib.ttf" if bold else "segoeui.ttf") if latin else ("msyhbd.ttc" if bold else "msyh.ttc"))
    return ImageFont.truetype(str(FONT_FOLDER/file),size)


def type_at(d,x,y,text,size,color=WHITE,bold=False,anchor="lm",latin=False):
    d.text((x,y),text,font=font(size,bold,latin),fill=color,anchor=anchor,stroke_width=0)


class FilmDeck(Switchboard):
    """Same vector artwork as the Tk deck; render at design resolution, then downsample."""
    def __init__(self,home):
        self.knobs={}
        self.art=Image.new("RGB",(900,460),BG); self.ink=ImageDraw.Draw(self.art)
        self.set_camera(home); self.scenery()
        self.background=self.art

    def rect(self,l,t,r,b,fill="",edge="",w=1):
        self.ink.rectangle((l,t,r,b),fill=fill or None,outline=edge or None,width=max(1,round(w)))

    def oval(self,x,y,rx,ry,fill="",edge="",w=1):
        self.ink.ellipse((x-rx,y-ry,x+rx,y+ry),fill=fill or None,outline=edge or None,width=max(1,round(w)))

    def line(self,*pts,ink=EDGE,w=1,dash=None):
        w=max(1,round(w))
        if not dash:
            self.ink.line(pts,fill=ink,width=w,joint="curve"); return
        for j in range(0,len(pts)-2,2):
            x,y,ex,ey=pts[j:j+4]; dx,dy=ex-x,ey-y
            n=hypot(dx,dy)
            for s in range(0,int(n),sum(dash)):
                z=min(n,s+dash[0])
                self.ink.line((x+s*dx/n,y+s*dy/n,x+z*dx/n,y+z*dy/n),fill=ink,width=w)

    def poly(self,*pts,fill,edge="",w=1):
        self.ink.polygon(pts,fill=fill)
        if edge: self.ink.line((*pts,*pts[:2]),fill=edge,width=max(1,round(w)),joint="curve")

    def txt(self,x,y,text,size=15,col=WHITE,bold=False,anchor="center"):
        type_at(self.ink,x,y,text,size,col,bold,{"center":"mm","w":"lm","e":"rm"}[anchor])

    def film(self,view,time):
        self.art=self.background.copy(); self.ink=ImageDraw.Draw(self.art)
        self.draw_lanes(view,time); self.controls(view,view.soundings)
        return self.art.crop((2,46,898,450)).resize((496,224),Image.Resampling.LANCZOS)


def stage(t):
    if t<2: return 0,"opening",0.
    if t>=30: return 1,"closing",t-30
    pair=int((t-2)//14); u=(t-2)%14
    for stop,begin,name in ((2.4,0,"probe"),(4.4,2.4,"seal0"),(6.4,4.4,"seal1"),
                            (9.3,6.4,"run0"),(12.2,9.3,"run1"),(14,12.2,"pair")):
        if u<stop: return pair,name,u-begin
    raise ValueError(u)


class Visible:
    lesson=False
    mode="watch"

    def __init__(self,row,frames,mark,stage_name,age):
        self.row,self.frames,self.name=row,frames,stage_name
        self.trip=SimpleNamespace(deal=SimpleNamespace(**row["deal"]),rig=SimpleNamespace(mark=mark))
        self.ready=stage_name not in ("opening","probe")
        self.axis=self.selected=self.voyage=None
        self.p=0.; self.elapsed=age; self.phase="deck"; self.raw=frames[0]
        self.echo=horizontal_centres(frames[1])-horizontal_centres(frames[0])
        self.soundings=({self.trip.deal.probe:SimpleNamespace(trace=self.echo)} if self.ready else {})
        self.idx=1 if stage_name in ("seal1","run1","pair","closing") else 0
        self.action=row["actions"][self.idx]
        self.actual=None
        if stage_name=="probe":
            self.axis,self.selected="x",self.trip.deal.probe
            self.p=smooth((age-.16)/.65)*(1-smooth((age-1.32)/.63))
            if self.p>=.999: self.raw=frames[1]
        elif stage_name.startswith("seal"):
            self.axis="y"; self.selected=self.action["side"] if age>=.25 else None
            self.p=smooth((age-.32)/.76)
        elif stage_name.startswith("run") or stage_name in ("pair","closing"):
            self.axis,self.selected="y",self.action["side"]
            self.phase="reveal"
            if stage_name in ("pair","closing"): self.elapsed=4.
            self.p=smooth((self.elapsed-.24)/1.05)
            self.voyage=SimpleNamespace(motion=np.array(self.action["motion"]),hits=self.action["hits"],won=self.action["passed"])
            if self.p>=.999:
                self.actual=self.voyage; self.raw=frames[2+self.idx]

    def goal(self): return self.action["goal"] if self.ready else None
    def arrival(self): return self.actual
    def show_gear(self): return self.ready
    def track_axes(self,side): return (side==self.trip.deal.probe,False) if not self.ready else (False,True)

    def status(self):
        if self.name=="opening": return "AT BERTH    /    等待试探",MUTED
        if self.name=="probe": return f"PROBE {HANDLE[self.selected]}    /    横试后复位",GOLD
        if self.name in ("pair","closing"):
            a,b=self.row["actions"]
            return f"{HANDLE[a['side']]}  {a['hits']}/3     +     {HANDLE[b['side']]}  {b['hits']}/3      |      "+("PASS  双过" if self.row["paired_pass"] else "MISS  未双过"),SEA if self.row["paired_pass"] else RED
        if self.selected is None: return "GOAL OPEN    /    发卡",MUTED
        side=HANDLE[self.selected]
        if self.name.startswith("seal"): return f"SELECT  {side}    /    指令封存",GOLD if side=="A" else BLUE
        if self.actual is None: return f"PULL  {side}    /    执行中",WHITE
        return f"{side}    {self.actual.hits}/3    /    "+("PASS  到泊" if self.actual.won else "MISS  未过"),SEA if self.actual.won else RED


STAGES={
    "opening":("One rig. Six frozen controllers.","每张卡三个目标泊位，送对两个就过。"),
    "probe":("Probe A, then reset.","先横试 A；四舱回到同一起点后才发卡。"),
    "seal0":("Seal the outbound choice.","选去程把手。先不执行，看不到结果。"),
    "seal1":("Seal the opposite choice.","目标反向。仍使用刚才的试探，不看去程结果。"),
    "run0":("Execute outbound from home.","从起点执行去程，按真实相机判定。"),
    "run1":("Execute return from the same home.","从同一起点另行执行返航。"),
    "pair":("Both cards must pass.","两张卡各至少 2/3 才算这一对通过。"),
    "closing":("Examples, not benchmark rankings.","只展示同一布局两对实操；总体成绩见论文表 2。"),
}


def background(k):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    type_at(d,32,26,"JSTP   /   FIELD RECORDINGS     01",16,SEA,True,latin=True)
    type_at(d,32,62,"Four-Track Harbor",42,WHITE,True,latin=True)
    type_at(d,418,66,"四轨货运：冻结模型同场实操",22,MUTED)
    d.rounded_rectangle((1204,18,1568,81),10,fill=BOX,outline=EDGE,width=2)
    type_at(d,1221,49,f"{k:,}",32,GOLD,True,latin=True)
    type_at(d,1549,38,"ACTION LABELS",16,WHITE,True,"rm",True)
    type_at(d,1549,60,"动作示范数，不是训练轮数",15,MUTED,False,"rm")
    for j,arm in enumerate(ARMS):
        x=32+j%3*518; y=139+j//3*295
        d.rounded_rectangle((x,y,x+501,y+285),9,fill=BOX,outline=EDGE,width=2)
        type_at(d,x+13,y+19,f"0{j+1}",15,SEA,True,latin=True)
        type_at(d,x+53,y+19,NAMES[arm],24,WHITE,True,latin=True)
    d.line((32,735,1568,735),fill=EDGE,width=2)
    type_at(d,212,761,"JSTP • RAW CAMERA / 原始相机",15,SEA,True)
    type_at(d,212,791,"Camera features + public instruments + goal → A/B. The simulator pulls the handle; no OS mouse.",23,WHITE,True,latin=True)
    type_at(d,212,826,"输入：相机特征、公开仪表、目标卡。输出：A / B，模拟器执行；测试不给隐藏接线、真值坐标或正确把手。",20,WHITE)
    type_at(d,32,877,"LAYOUT 113670000   /   TRAINING SEED 9301   /   TWO PUBLISHED GOAL PAIRS   /   SYNCHRONIZED PLAYBACK — NOT INFERENCE SPEED",16,MUTED,False,latin=True)
    return im


def make(k,book,cameras,out):
    rows={r["arm"]:r for r in book["runs"] if r["budget"]==k}
    shots={(a,i):cameras[f"{a}-{k}-pair{i+1}"] for a in ARMS for i in range(2)}
    deck=FilmDeck(shots["joint",0][0]); mark=Rig(rows["joint"]["pairs"][0]["deal"]["layout"]).mark
    bg=background(k)
    target=out/f"four-track-{k}.mp4"
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
         "-s:v",f"{W}x{H}","-r",str(FPS),"-i","pipe:0","-an","-c:v","libx264","-preset","fast",
         "-crf","23","-pix_fmt","yuv420p","-movflags","+faststart","-metadata",
         f"title=Four-Track Harbor | {k:,} action labels | Ke Yue",str(target)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for n in range(SECONDS*FPS):
            t=n/FPS; pair,name,age=stage(t); en,zh=STAGES[name]
            im=bg.copy(); d=ImageDraw.Draw(im)
            type_at(d,32,110,f"PAIR {pair+1}/2",16,SEA,True,latin=True)
            type_at(d,132,110,en,23,WHITE,True,latin=True)
            type_at(d,705,111,zh,19,MUTED)
            cache={}
            for j,arm in enumerate(ARMS):
                row=rows[arm]["pairs"][pair]
                v=Visible(row,shots[arm,pair],mark,name,age)
                # Identical camera bytes and actual selected action yield the same view.
                key=(tuple(row["camera_sha256"]),v.selected,v.idx)
                if key not in cache: cache[key]=deck.film(v,t)
                x=32+j%3*518; y=139+j//3*295
                im.paste(cache[key],(x+3,y+37))
                status,ink=v.status()
                type_at(d,x+12,y+271,status,16,ink,True)
                if arm=="joint":
                    im.paste(Image.fromarray(v.raw).resize((160,107),Image.Resampling.NEAREST),(32,749))
            d.line((32,893,1568,893),fill=EDGE,width=2)
            d.line((32,893,32+1536*n/(SECONDS*FPS-1),893),fill=SEA,width=2)
            if n==int(24.1*FPS): im.resize((1000,562),Image.Resampling.LANCZOS).save(out/f"poster-{k}.jpg",quality=86)
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError("FFmpeg failed: "+str(target))
    print("FILM",target,target.stat().st_size,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("folder",type=Path)
    p.add_argument("--fonts",type=Path,help="Directory containing Microsoft YaHei and Segoe UI")
    p.add_argument("--budget",type=int,choices=BUDGETS)
    a=p.parse_args()
    if a.fonts:
        global FONT_FOLDER
        FONT_FOLDER=a.fonts
    book=json.loads((a.folder/"runs.json").read_text(encoding="utf8"))
    with np.load(a.folder/"camera.npz",allow_pickle=False) as cameras:
        for k in ([a.budget] if a.budget else BUDGETS): make(k,book,cameras,a.folder)


if __name__=="__main__":
    main()
