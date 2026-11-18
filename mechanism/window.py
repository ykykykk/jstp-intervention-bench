"""完全独立的虚拟实验窗。只读训练日志/冻结权重，不收集桌面、不发送系统鼠标指令。"""
import argparse,json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image,ImageTk,ImageDraw
import torch
from .data import CFG,R,apparatus,episodes,context,images,act,ROOT,sha
from .net import Core,electrical
from jstp.arena import scene

class View:
    def __init__(self,root,release):
        self.root=root;self.rel=release;self.model=None;self.loaded=None;self.images=[];self.i=0
        root.title("JSTP · 联合未来实验台 | 岳轲");root.geometry("970x570");root.configure(bg="#eef3ef")
        bar=tk.Frame(root,bg="#143c38",pady=13);bar.pack(fill="x")
        tk.Label(bar,text="联合未来实验台　/　预测、驱动与实测",fg="white",bg="#143c38",font=("Microsoft YaHei UI",16,"bold")).pack(anchor="w",padx=18)
        self.status=tk.StringVar(); ttk.Label(root,textvariable=self.status).pack(anchor="w",padx=18,pady=8)
        self.canvas=tk.Canvas(root,width=936,height=275,bg="white",highlightthickness=0);self.canvas.pack()
        row=ttk.Frame(root);row.pack(fill="x",padx=18,pady=8)
        self.wire=tk.IntVar();self.goal=tk.IntVar();self.passive=tk.IntVar()
        for var,label in [(self.wire,"换挂接"),(self.goal,"换目标"),(self.passive,"外力牵引")]:
            ttk.Checkbutton(row,text=label,variable=var,command=self.draw).pack(side="left",padx=5)
        ttk.Button(row,text="下一个布局",command=self.next).pack(side="right")
        self.info=tk.StringVar();ttk.Label(root,textvariable=self.info,wraplength=925).pack(anchor="w",padx=18)
        self.graph=tk.Canvas(root,width=936,height=105,bg="#f7faf8",highlightthickness=0);self.graph.pack(pady=8)
        self.update()
    def next(self):self.i+=1;self.draw()
    def update(self):
        models=self.rel/"models";files=sorted(models.glob("*/curve.jsonl"),key=lambda p:p.stat().st_mtime)
        metas=sorted(models.glob("joint_symbol-*/freeze.json"))
        frozen=len(list(models.glob("*/freeze.json")))
        if metas and self.loaded!=metas[0]:
            p=metas[0]; self.model=Core().eval();f=p.parent/"last.pt"
            if sha(f)!=json.loads(p.read_text())["weights"]:raise ValueError("展示权重校验不符")
            self.model.load_state_dict(torch.load(f,map_location="cpu",weights_only=True));self.loaded=p
        self.graph.delete("all")
        if files:
            f=files[-1];lines=f.read_bytes().split(b'\n')[:-1]; curve=[json.loads(s) for s in lines]
            if curve:
                self.status.set(f"训练：{f.parent.name}　{curve[-1]['step']}/{CFG['steps']} 更新　|　冻结 {frozen}/15　|　曲线：验证集记录运动一致率，不是论文考试成绩")
                for a,b in zip(curve,curve[1:]):
                    pt=lambda c:(25+c['step']/CFG['steps']*885,95-85*c['validation_recorded_drive'])
                    self.graph.create_line(*pt(a),*pt(b),width=3,fill="#008572")
        self.draw();self.root.after(3500,self.update)
    @torch.inference_mode()
    def draw(self):
        official=(self.rel/"observed"/"after.json").exists()
        seed=(CFG["test_familiar"][0] if official else 40000000)+self.i%160
        s=scene(seed);bank=apparatus(seed,1,False);r=episodes(bank,seed);j=self.wire.get();g=self.goal.get();pa=self.passive.get()
        row=r[(r[:,1]==j)&(r[:,2]==0)&(r[:,3]==1)&(r[:,4]==g)&(r[:,5]==pa)][:1]
        b,p=images(s,j); canvas=self.canvas;canvas.delete("all");self.images=[]
        def tile(x,arr,label):
            q=Image.fromarray(arr).resize((219,146),Image.Resampling.NEAREST); photo=ImageTk.PhotoImage(q)
            self.images.append(photo);canvas.create_image(x,48,anchor="nw",image=photo);canvas.create_text(x+109,22,text=label,fill="#154840",font=("Microsoft YaHei UI",11))
        tile(9,p,"先前横向试探");tile(241,b,"复位后的现实")
        if self.model is None:self.info.set("等待首份冻结权重；不使用演示脚本替模型做决定。");return
        x=torch.tensor(context(bank,row));ctx,h,r0,vis,cmd=self.model.operation(x,"joint")
        u=int(electrical(cmd).item());ext=int(row[0,6]);aa=act(s,j,u,ext)
        z=vis[0,:R].numpy().reshape(8,12);heat=np.stack((z*.45,z,z*.8),-1)*255
        tile(473,heat.astype("uint8"),"预演驱动的感觉预测（粗网格）");tile(705,aa,"落实正式驱动后的相机")
        win=np.any(b[:43,32:64]!=aa[:43,32:64])==bool(g) and (u==0 if pa else u!=0)
        self.info.set(("冻结考试布局" if official else "不计分的新演示布局")+" "+str(seed)+
                      f"　目标：中带{'移动' if g else '保持'}；预演 {int(electrical(r0).item()):+d} → 自身 {u:+d}，外力 {ext:+d}。实测：{'达到' if win else '未达到'}目标。上图显示任务台；配对实验另有同构的无关台。")
        canvas.create_text(460,224,text="同一网络预测自身和感觉；虚拟控制器只落实有符号驱动，不对备选后果打分。",fill="#476460",font=("Microsoft YaHei UI",10))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--release",type=Path,default=ROOT/"evidence/mechanism");args=a.parse_args()
    torch.set_num_threads(2);root=tk.Tk();View(root,args.release);root.mainloop()
