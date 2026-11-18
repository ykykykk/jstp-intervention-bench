"""迟到目标实验的独立窗口：不接管系统鼠标，不播放归档影片。"""
import argparse,json,tkinter as tk
from pathlib import Path
from tkinter import ttk
import numpy as np
import torch
from PIL import Image,ImageTk
from .rig import C,REL,Rig,TRAIN,EXAM,lane_travel,goal_wins,read,dump
from .control import MODES,fingerprint,load_net,collect,decisions

GREEN="#17634f";BLUE="#245a77";ORANGE="#af582e";INK="#1b3333"
def side(x):return "A" if x<0 else "B"
class Window:
    def __init__(self,root,rel):
        self.root=root;self.rel=rel;self.net=None;self.basis=None;self.i=0;self.goal=0;self.after=False;self.ctx=None
        self.generation=0;self.caption=""
        root.title("JSTP · 先试探，后给目标 | 岳轲");root.configure(bg="#eef3f0")
        tk.Label(root,text="先试探，后给目标",font=("Microsoft YaHei UI",20,"bold"),fg=INK,bg="#eef3f0").pack(anchor="w",padx=19,pady=(12,0))
        self.status=tk.StringVar(value="等候训练日志");ttk.Label(root,textvariable=self.status).pack(anchor="w",padx=20,pady=5)
        self.graph=tk.Canvas(root,width=990,height=74,bg="white",highlightthickness=0);self.graph.pack(padx=18)
        self.canvas=tk.Canvas(root,width=990,height=251,bg="white",highlightthickness=0);self.canvas.pack(padx=18,pady=(7,0))
        self.explain=tk.StringVar();ttk.Label(root,textvariable=self.explain,wraplength=960).pack(anchor="w",padx=20,pady=8)
        bar=ttk.Frame(root);bar.pack(fill="x",padx=18,pady=5)
        self.mode=tk.IntVar(value=0)
        for t,k in [("原预测",0),("只改物理预期",2),("等幅无关改动",3)]:
            ttk.Radiobutton(bar,text=t,value=k,variable=self.mode,command=self.manual_draw).pack(side="left",padx=3)
        ttk.Button(bar,text="只换成相反目标",command=self.opposite).pack(side="left",padx=8)
        ttk.Button(bar,text="下一组目标",command=self.next_goal).pack(side="left")
        ttk.Button(bar,text="论文实例",command=self.example).pack(side="left",padx=5)
        ttk.Button(bar,text="连看三次实做",command=self.showcase).pack(side="left")
        ttk.Button(bar,text="换布局并重新试探",command=self.next).pack(side="right")
        self.score=tk.StringVar();ttk.Label(root,textvariable=self.score,wraplength=960).pack(anchor="w",padx=20,pady=(4,15))
        self.tick()
    def manual_draw(self):self.generation+=1;self.caption="";self.draw()
    def next(self):self.i+=1;self.ctx=None;self.manual_draw()
    def opposite(self):self.goal^=1;self.manual_draw()
    def next_goal(self):self.goal=(self.goal//2+1)*2;self.manual_draw()
    def example(self):
        if self.after:self.i=0;self.ctx=None;self.goal=4;self.mode.set(0);self.manual_draw()
    def showcase(self,record=None):
        if not self.after or self.basis is None:return
        self.generation+=1;stamp=self.generation;self.i=0;self.ctx=None;rows=[]
        steps=[(4,0,"① 原目标，正常执行。"),(5,0,"② 只换目标，同一经历再执行。"),(4,2,"③ 恢复原目标，只改内部后果；现实没有换。")]
        def advance(k):
            if self.generation!=stamp:return
            self.goal,mode,self.caption=steps[k];self.mode.set(mode);rows.append(self.draw())
            if k<2:self.root.after(2100,lambda:advance(k+1))
            elif record is not None:dump(record,dict(model=f"joint-{C['seeds'][0]}",kind="live window, same cached pregoal prefix, real virtual actuator",trials=rows))
        advance(0)
    def tick(self):
        files=sorted((self.rel/"models").glob("*/curve.jsonl"),key=lambda p:p.stat().st_mtime)
        frozen=len(list((self.rel/"models").glob("*/freeze.json")))
        self.graph.delete("all")
        if files:
            l=files[-1];rows=[json.loads(x) for x in l.read_bytes().split(b'\n')[:-1]]
            if rows:
                self.status.set(f"训练 {l.parent.name} · {rows[-1]['step']} / {C['updates']} 步　｜　已冻结 {frozen}/9 份　｜　下图是训练目标支持上的验证正确率，不是考试成绩")
                for a,b in zip(rows,rows[1:]):
                    pt=lambda e:(17+950*e['step']/C['updates'],63-50*e['validation'])
                    self.graph.create_line(*pt(a),*pt(b),fill=GREEN,width=3)
        name=f"joint-{C['seeds'][0]}"
        if self.net is None and (self.rel/"models"/name/"freeze.json").exists():self.net=load_net(self.rel,"joint",C['seeds'][0])
        if self.basis is None and (self.rel/"calibrated.json").exists():self.basis=read(self.rel/"calibration"/(name+".npz"))
        official=(self.rel/"after.json").exists()
        if official!=self.after:self.after=official;self.ctx=None;self.goal=4 if official else 0
        stat=self.rel/"analysis/summary.json"
        if stat.exists():
            m=json.loads(stat.read_text(encoding="utf8"))["arms"]["joint"]
            self.score.set(f"冻结批量成绩（不是当前单例）：两个新目标同时做对 {m['pair']['value']:.1%}；应换手题按供体换手 {m['directed']['value']:.1%}；不应换手题误换 {m['false_flip']['value']:.1%}。")
        self.draw();self.root.after(3500,self.tick)
    def draw(self):
        can=self.canvas;can.delete("all");self.photos=[]
        if self.net is None:can.create_text(495,125,text="等候首份冻结权重；不以规则替模型操作。",fill=INK);return
        if self.ctx is None:
            r=Rig((C['exam'][0] if self.after else C['demo_start'])+self.i%192)
            c=collect(self.net,[r]);self.ctx=r,c
        r,c=self.ctx;p=int(c['p'][0]);goals=EXAM if self.after else TRAIN;self.goal%=len(goals)
        g=goals[self.goal];mode=self.mode.get() if self.basis is not None else 0
        z=decisions(self.net,"joint",c,g[None],self.basis);base=int(z['u'][0,0,0,0]);u=int(z['u'][0,0,0,mode])
        w=r.world(0);peak=w.probe(p);w.set_clutch(0);end=w.stroke(u);ok=bool(goal_wins(lane_travel(r.base,end),g))
        def title(x,t,col):can.create_text(x+116,18,text=t,fill=col,font=("Microsoft YaHei UI",10,"bold"))
        def tile(x,im):
            ph=ImageTk.PhotoImage(Image.fromarray(im).resize((228,152),Image.Resampling.NEAREST))
            self.photos.append(ph);can.create_image(x,38,anchor="nw",image=ph)
        title(10,"① 现场横向试探 · "+side(p),BLUE);tile(10,peak)
        title(258,"② 复位 · 此后才给目标",BLUE);tile(258,r.base)
        title(502,"③ 同把手纵向后果 · 内部读出",ORANGE)
        for j,(old,new) in enumerate(zip(z['eye'][0,0,0],z['eye'][0,0,mode])):
            x=511+j*18;can.create_line(x,112,x,112+float(np.clip(old,-1.5,1.5))*51,fill="#b1c6be",width=5)
            can.create_line(x+5,112,x+5,112+float(np.clip(new,-1.5,1.5))*51,fill=ORANGE,width=5)
        can.create_line(505,112,732,112,fill="#869791");can.create_text(618,200,text="灰绿：原预期　橙：当前预期",fill=ORANGE)
        title(748,"④ 在原机构实际执行 · "+side(u),GREEN);tile(748,end)
        req="  ".join(f"{j+1}{'↓' if k>0 else '↑'}" for j,k in enumerate(g) if k)
        can.create_text(19,226,anchor="w",text=f"目标卡　{req}　｜　所指定的方向过半即成功　｜　本次{'成功' if ok else '失败'}",fill=GREEN if ok else ORANGE,font=("Microsoft YaHei UI",12,"bold"))
        self.explain.set(self.caption+f"{'冻结考试' if self.after else '非考试演示'}布局 {r.seed}；保持同一真实经历和离合设置。原预测操作 {side(base)}，当前操作 {side(u)}。换目标时复用目标到达前的同一隐状态；干预只改这段隐式后果。"+("" if self.basis is not None else "物理方向尚未校准，目前显示原预测。"))
        return dict(layout=r.seed,probe=p,goal=g.tolist(),mode=MODES[mode],prefix=z["prefix"][0,0].decode(),
                    command=u,success=ok,actual_camera_sha256=fingerprint(end).decode())
def main():
    a=argparse.ArgumentParser();a.add_argument("--release",type=Path,default=REL);a.add_argument("--showcase",action="store_true")
    a.add_argument("--record",type=Path,help="把三次现场实做写入新文件，不覆盖既有档案");args=a.parse_args()
    if args.record is not None and (not args.showcase or args.record.exists()):a.error("--record 须与 --showcase 同用，并且只能写新文件")
    torch.set_num_threads(2);root=tk.Tk();w=Window(root,args.release)
    if args.showcase:
        if not w.after:raise RuntimeError("现场三次演示要求正式物理考试已经完成")
        root.after(650,lambda:w.showcase(args.record))
    root.mainloop()
if __name__=="__main__":main()
